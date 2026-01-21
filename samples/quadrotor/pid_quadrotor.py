"""
Quadrotor PID制御によるホバリング

学習は不要です。PIDパラメータを調整するだけで動作します。
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# ============================================================================
# PID制御器クラス
# ============================================================================

class PIDController:
    """PID制御器"""
    def __init__(self, kp, ki, kd, integral_limit=10.0):
        self.kp = kp  # 比例ゲイン
        self.ki = ki  # 積分ゲイン
        self.kd = kd  # 微分ゲイン
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0
    
    def compute(self, target, current, current_vel, dt):
        """PID制御の出力を計算"""
        error = target - current
        
        # 積分項を更新（ウィンドウアップ防止）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # PID制御: 出力 = Kp × 誤差 + Ki × 積分誤差 + Kd × 誤差の微分
        output = (self.kp * error + 
                  self.ki * self.integral + 
                  self.kd * (-current_vel))  # 速度は誤差の微分の負
        
        self.last_error = error
        return output
    
    def reset(self):
        """積分項をリセット"""
        self.integral = 0.0


# ============================================================================
# Quadrotor制御クラス
# ============================================================================

class QuadrotorController:
    """QuadrotorのPID制御"""
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # プロペラの位置（URDFから取得した値）
        # prop1: 右前, prop2: 前, prop3: 左前, prop4: 後
        self.prop_positions = [
            np.array([0.175, 0, 0]),      # prop1 (右)
            np.array([0, 0.175, 0]),      # prop2 (前)
            np.array([-0.175, 0, 0]),     # prop3 (左)
            np.array([0, -0.175, 0])      # prop4 (後)
        ]
        
        # プロペラのモーメント係数（Yaw制御用）
        self.prop_moments = [0.0245, -0.0245, 0.0245, -0.0245]
        
        # プロペラ間の距離（Roll/Pitch制御用）
        self.arm_length = 0.175
        
        # PID制御器を初期化
        # 高さ制御
        self.height_pid = PIDController(kp=15.0, ki=2.0, kd=8.0)
        
        # 姿勢制御（Roll, Pitch, Yaw）
        self.roll_pid = PIDController(kp=10.0, ki=0.5, kd=5.0)
        self.pitch_pid = PIDController(kp=10.0, ki=0.5, kd=5.0)
        self.yaw_pid = PIDController(kp=5.0, ki=0.1, kd=2.0)
        
        # 目標値
        self.target_height = 2.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
        # 基本推力（重力を打ち消す）
        self.mass = 0.5  # kg (URDFから)
        self.gravity = 9.81
        self.base_thrust = self.mass * self.gravity / 4.0  # 各プロペラの基本推力
        
        # 最大推力
        self.max_thrust = 20.0  # N (各プロペラ)
        self.min_thrust = 0.0   # N
    
    def update(self, dt):
        """
        制御を更新してプロペラの推力を計算
        
        Returns:
            list: 4つのプロペラの推力 [prop1, prop2, prop3, prop4]
        """
        # 現在の状態を取得
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # オイラー角に変換
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        
        # 速度
        vel_z = vel[2]
        roll_vel = ang_vel[0]
        pitch_vel = ang_vel[1]
        yaw_vel = ang_vel[2]
        
        # PID制御で各軸の制御出力を計算
        height_output = self.height_pid.compute(self.target_height, pos[2], vel_z, dt)
        roll_output = self.roll_pid.compute(self.target_roll, roll, roll_vel, dt)
        pitch_output = self.pitch_pid.compute(self.target_pitch, pitch, pitch_vel, dt)
        yaw_output = self.yaw_pid.compute(self.target_yaw, yaw, yaw_vel, dt)
        
        # 各プロペラの推力を計算
        # 高さ制御: 全プロペラに同じ推力
        thrust_height = self.base_thrust + height_output / 4.0
        
        # Roll制御: prop1とprop3の差（右と左）
        # Rollが正（右に傾く）→ prop1を増やし、prop3を減らす
        # トルク = 力 × 距離 → 力 = トルク / 距離
        thrust_roll = roll_output / (2.0 * self.arm_length)
        
        # Pitch制御: prop2とprop4の差（前と後）
        # Pitchが正（前傾）→ prop2を増やし、prop4を減らす
        thrust_pitch = pitch_output / (2.0 * self.arm_length)
        
        # Yaw制御: モーメントの符号に基づいて配分
        # 正のYaw（時計回り）→ 正のモーメントを持つプロペラを減らす
        # モーメント係数で正規化
        yaw_scale = 0.1  # Yaw制御のスケール
        thrust_yaw = yaw_output * yaw_scale
        
        # 各プロペラの推力
        # prop1: 右, prop2: 前, prop3: 左, prop4: 後
        thrusts = [
            thrust_height + thrust_roll + thrust_yaw * self.prop_moments[0],  # prop1 (右)
            thrust_height + thrust_pitch + thrust_yaw * self.prop_moments[1],  # prop2 (前)
            thrust_height - thrust_roll + thrust_yaw * self.prop_moments[2],   # prop3 (左)
            thrust_height - thrust_pitch + thrust_yaw * self.prop_moments[3]   # prop4 (後)
        ]
        
        # 推力の制限
        thrusts = [np.clip(t, self.min_thrust, self.max_thrust) for t in thrusts]
        
        return thrusts, {
            'height': pos[2],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'height_error': self.target_height - pos[2],
            'roll_error': self.target_roll - roll,
            'pitch_error': self.target_pitch - pitch,
            'yaw_error': self.target_yaw - yaw
        }
    
    def apply_thrusts(self, thrusts):
        """プロペラの推力を適用"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 各プロペラの位置に個別に力を適用
        # これにより、推力の差が自動的にトルクを生成します
        for i, (thrust, prop_pos_local) in enumerate(zip(thrusts, self.prop_positions)):
            # ローカル座標系での力（上向き）
            force_local = [0, 0, thrust]
            
            # ワールド座標系に変換
            force_world = p.rotateVector(orn, force_local)
            # リストに変換（タプルの場合があるため）
            if isinstance(force_world, tuple):
                force_world = list(force_world)
            
            # プロペラの位置をワールド座標系に変換
            # prop_pos_localをリストに変換（numpy配列の場合があるため）
            prop_pos_local_list = list(prop_pos_local) if isinstance(prop_pos_local, np.ndarray) else prop_pos_local
            
            prop_pos_world, _ = p.multiplyTransforms(
                pos, orn,
                prop_pos_local_list, [0, 0, 0, 1]
            )
            # リストに変換（タプルの場合があるため）
            if isinstance(prop_pos_world, tuple):
                prop_pos_world = list(prop_pos_world)
            
            # 各プロペラの位置に力を適用
            p.applyExternalForce(
                self.robot_id,
                -1,  # ベースリンク
                force_world,
                prop_pos_world,
                p.WORLD_FRAME
            )
        
        # Yaw制御用のトルク（モーメントの差）
        # これは各プロペラの回転による反トルクです
        yaw_torque = sum(t * m for t, m in zip(thrusts, self.prop_moments))
        
        # Yawトルクを適用（Z軸周りの回転）
        if abs(yaw_torque) > 0.001:  # 小さな値は無視
            # ローカル座標系でのYawトルク
            torque_local = [0, 0, yaw_torque]
            
            # ワールド座標系に変換
            torque_world = p.rotateVector(orn, torque_local)
            
            # トルクを適用（別の方法を試す）
            # applyExternalTorqueが使えない場合、各プロペラに接線方向の力を適用
            # ただし、これは複雑なので、まずはYaw制御なしで試す
            pass  # Yaw制御は後で実装
    
    def reset(self):
        """PID制御器をリセット"""
        self.height_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()


# ============================================================================
# メイン関数
# ============================================================================

def main():
    """メイン関数"""
    print("=" * 60)
    print("🚁 Quadrotor PID制御によるホバリング")
    print("=" * 60)
    print("\n💡 学習は不要です。PIDパラメータを調整するだけで動作します。")
    print("=" * 60)
    
    # PyBulletに接続
    print("\n🚀 PyBulletをGUIモードで起動します...")
    device_id = p.connect(p.GUI)
    if device_id < 0:
        print("❌ GUIモードでの接続に失敗しました")
        exit(1)
    
    # パス設定
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    quadrotor_dir = os.path.join(project_root, "Quadrotor")
    data_path = pybullet_data.getDataPath()
    
    p.setAdditionalSearchPath(data_path)
    p.setAdditionalSearchPath(quadrotor_dir)
    p.setGravity(0, 0, -9.81)
    
    # 床をロード
    plane_path = os.path.join(data_path, "plane.urdf")
    p.loadURDF(plane_path)
    print("✅ 床をロードしました")
    
    # Quadrotorをロード
    spawn_height = 1.0
    quadrotor_path = os.path.join(quadrotor_dir, "quadrotor.urdf")
    robot_id = p.loadURDF(quadrotor_path, [0, 0, spawn_height])
    print(f"✅ Quadrotorをロードしました（高さ: {spawn_height}m）")
    
    # 制御器を初期化
    controller = QuadrotorController(robot_id)
    controller.target_height = 2.0  # 目標高さ2m
    
    print(f"\n📊 制御パラメータ:")
    print(f"   目標高さ: {controller.target_height}m")
    print(f"   目標Roll: {controller.target_roll}rad")
    print(f"   目標Pitch: {controller.target_pitch}rad")
    print(f"   目標Yaw: {controller.target_yaw}rad")
    
    print(f"\n🎮 制御を開始します...")
    print(f"   GUIでドローンの動作を確認してください")
    print(f"   Ctrl+Cで終了\n")
    
    # シミュレーションループ
    dt = 1.0 / 240.0  # PyBulletの標準タイムステップ
    step_count = 0
    
    try:
        while True:
            # PID制御を更新
            thrusts, state = controller.update(dt)
            
            # 推力を適用
            controller.apply_thrusts(thrusts)
            
            # シミュレーションを1ステップ進める
            p.stepSimulation()
            
            # 状態を表示（1秒ごと）
            if step_count % 240 == 0:
                print(f"Time: {step_count * dt:.1f}s | "
                      f"高さ: {state['height']:.3f}m (誤差: {state['height_error']:.3f}m) | "
                      f"Roll: {np.degrees(state['roll']):.1f}° | "
                      f"Pitch: {np.degrees(state['pitch']):.1f}° | "
                      f"Yaw: {np.degrees(state['yaw']):.1f}° | "
                      f"推力: [{thrusts[0]:.2f}, {thrusts[1]:.2f}, {thrusts[2]:.2f}, {thrusts[3]:.2f}] N")
            
            step_count += 1
            time.sleep(dt)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  制御を終了します...")
    
    p.disconnect()
    print("✅ シミュレーションを終了しました")


if __name__ == "__main__":
    main()
