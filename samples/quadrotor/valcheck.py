"""
valcheck: Quadrotor速度検証スクリプト

pid_quadrotor.pyをベースに、ホバリング中の速度を詳細に検証する。
PyBulletから取得した速度と実際の位置変化から計算した速度を比較する。
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import logging
from datetime import datetime

# ============================================================================
# PID制御器クラス
# ============================================================================

class PIDController:
    """PID制御器"""
    def __init__(self, kp, ki, kd, integral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0
    
    def compute(self, target, current, current_vel, dt):
        """PID制御の出力を計算"""
        error = target - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        output = self.kp * error + self.ki * self.integral + self.kd * (-current_vel)
        self.last_error = error
        return output
    
    def reset(self):
        """積分項をリセット"""
        self.integral = 0.0


# ============================================================================
# Quadrotor制御クラス
# ============================================================================

class QuadrotorController:
    """QuadrotorのPID制御（pid_quadrotor.pyと同じ）"""
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.prop_positions = [
            np.array([0.175, 0, 0]),
            np.array([0, 0.175, 0]),
            np.array([-0.175, 0, 0]),
            np.array([0, -0.175, 0]),
        ]
        self.prop_moments = [0.0245, -0.0245, 0.0245, -0.0245]
        self.arm_length = 0.175
        
        self.height_pid = PIDController(kp=15.0, ki=2.0, kd=8.0)
        self.roll_pid = PIDController(kp=10.0, ki=0.5, kd=5.0)
        self.pitch_pid = PIDController(kp=10.0, ki=0.5, kd=5.0)
        self.yaw_pid = PIDController(kp=5.0, ki=0.1, kd=2.0)
        
        self.target_height = 2.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
        self.mass = 0.5
        self.gravity = 9.81
        self.base_thrust = self.mass * self.gravity / 4.0
        self.max_thrust = 20.0
        self.min_thrust = 0.0
    
    def update(self, dt):
        """制御を更新してプロペラの推力を計算"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        
        vel_z = vel[2]
        roll_vel = ang_vel[0]
        pitch_vel = ang_vel[1]
        yaw_vel = ang_vel[2]
        
        height_output = self.height_pid.compute(self.target_height, pos[2], vel_z, dt)
        roll_output = self.roll_pid.compute(self.target_roll, roll, roll_vel, dt)
        pitch_output = self.pitch_pid.compute(self.target_pitch, pitch, pitch_vel, dt)
        yaw_output = self.yaw_pid.compute(self.target_yaw, yaw, yaw_vel, dt)
        
        thrust_height = self.base_thrust + height_output / 4.0
        thrust_roll = roll_output / (2.0 * self.arm_length)
        thrust_pitch = pitch_output / (2.0 * self.arm_length)
        yaw_scale = 0.1
        thrust_yaw = yaw_output * yaw_scale
        
        thrusts = [
            thrust_height + thrust_roll + thrust_yaw * self.prop_moments[0],
            thrust_height + thrust_pitch + thrust_yaw * self.prop_moments[1],
            thrust_height - thrust_roll + thrust_yaw * self.prop_moments[2],
            thrust_height - thrust_pitch + thrust_yaw * self.prop_moments[3],
        ]
        thrusts = [np.clip(t, self.min_thrust, self.max_thrust) for t in thrusts]
        
        # 速度情報を返すために、ワールド座標系と機体座標系の速度を計算
        vel_x_world = vel[0]
        vel_y_world = vel[1]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vel_x_body = vel_x_world * cos_yaw + vel_y_world * sin_yaw
        vel_y_body = -vel_x_world * sin_yaw + vel_y_world * cos_yaw
        
        return thrusts, {
            'height': pos[2],
            'x': pos[0],
            'y': pos[1],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'vel_x_world': vel_x_world,
            'vel_y_world': vel_y_world,
            'vel_x_body': vel_x_body,
            'vel_y_body': vel_y_body,
            'height_error': self.target_height - pos[2],
        }
    
    def apply_thrusts(self, thrusts):
        """プロペラの推力を適用"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        for i, (thrust, prop_pos_local) in enumerate(zip(thrusts, self.prop_positions)):
            force_local = [0, 0, thrust]
            force_world = p.rotateVector(orn, force_local)
            if isinstance(force_world, tuple):
                force_world = list(force_world)
            prop_pos_local_list = list(prop_pos_local) if isinstance(prop_pos_local, np.ndarray) else prop_pos_local
            prop_pos_world, _ = p.multiplyTransforms(pos, orn, prop_pos_local_list, [0, 0, 0, 1])
            if isinstance(prop_pos_world, tuple):
                prop_pos_world = list(prop_pos_world)
            p.applyExternalForce(self.robot_id, -1, force_world, prop_pos_world, p.WORLD_FRAME)


# ============================================================================
# メイン関数
# ============================================================================

def main():
    """メイン関数"""
    print("=" * 60)
    print("valcheck: Quadrotor速度検証スクリプト")
    print("=" * 60)
    
    # PyBulletに接続
    device_id = p.connect(p.GUI)
    if device_id < 0:
        print("❌ PyBullet接続失敗")
        exit(1)
    
    # パス設定
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    quadrotor_dir = os.path.join(project_root, "Quadrotor")
    data_path = pybullet_data.getDataPath()
    p.setAdditionalSearchPath(data_path)
    p.setAdditionalSearchPath(quadrotor_dir)
    p.setGravity(0, 0, -9.81)
    
    # 床とQuadrotorをロード
    plane_path = os.path.join(data_path, "plane.urdf")
    p.loadURDF(plane_path)
    quadrotor_path = os.path.join(quadrotor_dir, "quadrotor.urdf")
    robot_id = p.loadURDF(quadrotor_path, [0, 0, 1.0])
    
    # 制御器を初期化
    controller = QuadrotorController(robot_id)
    controller.target_height = 2.0
    
    # ロギング設定
    log_dir = os.path.join(project_root, "samples", "quadrotor", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "valcheck.log")
    log_file_rotated = os.path.join(log_dir, "valcheck.log.1")
    
    # ログローテーション
    if os.path.exists(log_file):
        if os.path.exists(log_file_rotated):
            os.remove(log_file_rotated)
        os.rename(log_file, log_file_rotated)
    
    # ロガーを設定
    logger = logging.getLogger('valcheck')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    logger.info("=" * 60)
    logger.info("valcheck: Quadrotor速度検証スクリプト")
    logger.info("=" * 60)
    logger.info(f"ログファイル: {log_file}")
    logger.info("ホバリング中の速度を検証します（ピッチ前傾テスト含む）")
    
    # ピッチ前傾テストの設定
    PITCH_TEST_START_TIME = 5.0  # ピッチ前傾開始時刻 [s]
    PITCH_TEST_DURATION = 0.1     # ピッチ前傾持続時間 [s]
    PITCH_TEST_ANGLE = 0.05       # ピッチ前傾角度 [rad] ≈ 2.87°
    
    logger.info(f"ピッチ前傾テスト: t={PITCH_TEST_START_TIME}sから{PITCH_TEST_DURATION}s間、pitch={np.degrees(PITCH_TEST_ANGLE):.2f}°")
    
    print(f"ログファイル: {log_file}")
    print("ホバリングを開始します...")
    print(f"ピッチ前傾テスト: t={PITCH_TEST_START_TIME}sから{PITCH_TEST_DURATION}s間、pitch={np.degrees(PITCH_TEST_ANGLE):.2f}°")
    print("Ctrl+Cで終了\n")
    
    # シミュレーションループ
    dt = 1.0 / 240.0
    step = 0
    prev_x = None
    prev_y = None
    prev_t = None
    
    # n-1の状態を保存（制御フロー追跡用）
    prev_state_n_minus_1 = None
    prev_thrusts_n_minus_1 = None
    # n-2の状態を保存（Simpson法などの高度な検証用）
    prev_state_n_minus_2 = None
    prev_t_n_minus_1 = None
    prev_t_n_minus_2 = None
    # ログ間隔の間の各ステップの位置変化とyaw角度を記録（各ステップごとの座標変換用）
    step_positions = []  # [(dx, dy, yaw), ...]
    step_positions_prev_x = None
    step_positions_prev_y = None
    
    # 安定化のため、最初の3秒はログを取らない
    STABILIZATION_TIME = 3.0
    LOG_INTERVAL = 10  # 10ステップごと（約0.04秒、姿勢変化の影響を最小限に抑える）
    
    try:
        while True:
            t = step * dt
            
            # ピッチ前傾テスト: 指定時刻に少しだけピッチを前傾させ、すぐ戻す
            if PITCH_TEST_START_TIME <= t < PITCH_TEST_START_TIME + PITCH_TEST_DURATION:
                controller.target_pitch = PITCH_TEST_ANGLE
            else:
                controller.target_pitch = 0.0
            
            # n-1の状態を保存（制御フロー追跡用）
            # この時点で、前回のstepSimulation()後の状態が取得できる
            if step > 0:
                pos_n_minus_1, orn_n_minus_1 = p.getBasePositionAndOrientation(robot_id)
                vel_n_minus_1, ang_vel_n_minus_1 = p.getBaseVelocity(robot_id)
                euler_n_minus_1 = p.getEulerFromQuaternion(orn_n_minus_1)
                roll_n_minus_1, pitch_n_minus_1, yaw_n_minus_1 = euler_n_minus_1
                
                # 機体座標系への変換
                cos_yaw_n_minus_1 = np.cos(yaw_n_minus_1)
                sin_yaw_n_minus_1 = np.sin(yaw_n_minus_1)
                vel_x_world_n_minus_1 = vel_n_minus_1[0]
                vel_y_world_n_minus_1 = vel_n_minus_1[1]
                vel_x_body_n_minus_1 = vel_x_world_n_minus_1 * cos_yaw_n_minus_1 + vel_y_world_n_minus_1 * sin_yaw_n_minus_1
                vel_y_body_n_minus_1 = -vel_x_world_n_minus_1 * sin_yaw_n_minus_1 + vel_y_world_n_minus_1 * cos_yaw_n_minus_1
                
                prev_state_n_minus_1 = {
                    'x': pos_n_minus_1[0],
                    'y': pos_n_minus_1[1],
                    'z': pos_n_minus_1[2],
                    'vel_x_world': vel_x_world_n_minus_1,
                    'vel_y_world': vel_y_world_n_minus_1,
                    'vel_x_body': vel_x_body_n_minus_1,
                    'vel_y_body': vel_y_body_n_minus_1,
                    'roll': roll_n_minus_1,
                    'pitch': pitch_n_minus_1,
                    'yaw': yaw_n_minus_1,
                    'target_pitch': controller.target_pitch,
                    'target_roll': controller.target_roll,
                    'target_yaw': controller.target_yaw,
                }
                prev_t_n_minus_1 = t
            
            # 制御更新（n-1の状態に基づいて制御を決定）
            thrusts, state = controller.update(dt)
            
            # 制御出力を保存
            prev_thrusts_n_minus_1 = thrusts.copy()
            
            # 推力を適用
            controller.apply_thrusts(thrusts)
            
            # 物理シミュレーションを進める（n-1 → n）
            p.stepSimulation()
            
            step += 1
            
            # nの状態を取得（stepSimulation()後の状態）
            pos_n, orn_n = p.getBasePositionAndOrientation(robot_id)
            vel_n, ang_vel_n = p.getBaseVelocity(robot_id)
            euler_n = p.getEulerFromQuaternion(orn_n)
            roll_n, pitch_n, yaw_n = euler_n
            
            # 機体座標系への変換
            cos_yaw_n = np.cos(yaw_n)
            sin_yaw_n = np.sin(yaw_n)
            vel_x_world_n = vel_n[0]
            vel_y_world_n = vel_n[1]
            vel_x_body_n = vel_x_world_n * cos_yaw_n + vel_y_world_n * sin_yaw_n
            vel_y_body_n = -vel_x_world_n * sin_yaw_n + vel_y_world_n * cos_yaw_n
            
            state_n = {
                'x': pos_n[0],
                'y': pos_n[1],
                'z': pos_n[2],
                'vel_x_world': vel_x_world_n,
                'vel_y_world': vel_y_world_n,
                'vel_x_body': vel_x_body_n,
                'vel_y_body': vel_y_body_n,
                'roll': roll_n,
                'pitch': pitch_n,
                'yaw': yaw_n,
            }
            
            x, y, z = state_n['x'], state_n['y'], state_n['z']
            
            # ログ間隔の間の各ステップの位置変化とyaw角度を記録
            if step_positions_prev_x is not None and step_positions_prev_y is not None:
                step_dx = x - step_positions_prev_x
                step_dy = y - step_positions_prev_y
                step_positions.append((step_dx, step_dy, yaw_n))
            # 最初のステップでは、現在の位置を記録（次のステップから位置変化を記録）
            if step_positions_prev_x is None:
                step_positions_prev_x = x
                step_positions_prev_y = y
            else:
                step_positions_prev_x = x
                step_positions_prev_y = y
            
            # 安定化後、短い間隔で制御フローをロギング（姿勢変化の影響を最小限に抑えるため）
            # 注意: ログ間隔を固定（LOG_INTERVAL = 10ステップ = 約0.04秒）にして、等間隔を保証
            is_pitch_test = PITCH_TEST_START_TIME <= t < PITCH_TEST_START_TIME + PITCH_TEST_DURATION + 1.0  # テスト後1秒間も詳細ログ
            log_interval = LOG_INTERVAL  # 10ステップごと（等間隔を保証）
            
            if t >= STABILIZATION_TIME and step % log_interval == 0 and prev_state_n_minus_1 is not None:
                # n-1の状態（制御決定時の状態）
                logger.info(f"=== ループ n={step} (t={t:.2f}s) ===")
                logger.info(f"[n-1の状態] 制御決定時の状態:")
                logger.info(f"  位置: x={prev_state_n_minus_1['x']*100:.2f} y={prev_state_n_minus_1['y']*100:.2f} z={prev_state_n_minus_1['z']:.3f}")
                logger.info(f"  速度(ワールド): vel_x={prev_state_n_minus_1['vel_x_world']*100:.2f} vel_y={prev_state_n_minus_1['vel_y_world']*100:.2f} cm/s")
                logger.info(f"  速度(機体): vel_x_body={prev_state_n_minus_1['vel_x_body']*100:.2f} vel_y_body={prev_state_n_minus_1['vel_y_body']*100:.2f} cm/s")
                logger.info(f"  姿勢: r={np.degrees(prev_state_n_minus_1['roll']):.2f}° p={np.degrees(prev_state_n_minus_1['pitch']):.2f}° y={np.degrees(prev_state_n_minus_1['yaw']):.2f}°")
                logger.info(f"  目標値: target_pitch={np.degrees(prev_state_n_minus_1['target_pitch']):.2f}° target_roll={np.degrees(prev_state_n_minus_1['target_roll']):.2f}° target_yaw={np.degrees(prev_state_n_minus_1['target_yaw']):.2f}°")
                
                # 制御出力
                logger.info(f"[制御出力] n-1の状態に基づいて決定した推力:")
                logger.info(f"  thrusts: prop1={prev_thrusts_n_minus_1[0]:.4f} prop2={prev_thrusts_n_minus_1[1]:.4f} prop3={prev_thrusts_n_minus_1[2]:.4f} prop4={prev_thrusts_n_minus_1[3]:.4f} N")
                
                # nの状態（stepSimulation()後の状態）
                logger.info(f"[nの状態] stepSimulation()後の結果:")
                logger.info(f"  位置: x={state_n['x']*100:.2f} y={state_n['y']*100:.2f} z={state_n['z']:.3f}")
                logger.info(f"  速度(ワールド): vel_x={state_n['vel_x_world']*100:.2f} vel_y={state_n['vel_y_world']*100:.2f} cm/s")
                logger.info(f"  速度(機体): vel_x_body={state_n['vel_x_body']*100:.2f} vel_y_body={state_n['vel_y_body']*100:.2f} cm/s")
                logger.info(f"  姿勢: r={np.degrees(state_n['roll']):.2f}° p={np.degrees(state_n['pitch']):.2f}° y={np.degrees(state_n['yaw']):.2f}°")
                
                # 変化量
                dx = state_n['x'] - prev_state_n_minus_1['x']
                dy = state_n['y'] - prev_state_n_minus_1['y']
                dz = state_n['z'] - prev_state_n_minus_1['z']
                logger.info(f"[変化量] n-1 → n:")
                logger.info(f"  Δ位置: Δx={dx*100:+.2f} cm Δy={dy*100:+.2f} cm Δz={dz*100:+.2f} cm")
                logger.info(f"  Δ速度(ワールド): Δvel_x={(state_n['vel_x_world'] - prev_state_n_minus_1['vel_x_world'])*100:+.2f} Δvel_y={(state_n['vel_y_world'] - prev_state_n_minus_1['vel_y_world'])*100:+.2f} cm/s")
                logger.info(f"  Δ速度(機体): Δvel_x_body={(state_n['vel_x_body'] - prev_state_n_minus_1['vel_x_body'])*100:+.2f} Δvel_y_body={(state_n['vel_y_body'] - prev_state_n_minus_1['vel_y_body'])*100:+.2f} cm/s")
                
                # 位置変化から計算した速度（n-1からnへの平均速度）
                # 注意: これは複数ステップにわたる変化なので、実際の時間間隔を計算
                if prev_t is not None:
                    dt_actual = t - prev_t
                else:
                    dt_actual = dt * log_interval  # 初回はログ間隔を使用
                vel_x_from_pos = dx / dt_actual if dt_actual > 0 else 0.0
                vel_y_from_pos = dy / dt_actual if dt_actual > 0 else 0.0
                logger.info(f"[検証1] 位置変化から計算した速度（n-1→nの平均速度）:")
                logger.info(f"  vel_x={vel_x_from_pos*100:.2f} vel_y={vel_y_from_pos*100:.2f} cm/s (dt={dt_actual:.4f}s)")
                
                # n-1の速度とnの速度の平均（台形公式）
                vel_x_avg = (prev_state_n_minus_1['vel_x_world'] + state_n['vel_x_world']) / 2.0
                vel_y_avg = (prev_state_n_minus_1['vel_y_world'] + state_n['vel_y_world']) / 2.0
                logger.info(f"[検証2] n-1とnの速度の平均（台形公式）:")
                logger.info(f"  vel_x_avg={vel_x_avg*100:.2f} vel_y_avg={vel_y_avg*100:.2f} cm/s")
                
                # 加速度を考慮した検証
                accel_x = (state_n['vel_x_world'] - prev_state_n_minus_1['vel_x_world']) / dt_actual if dt_actual > 0 else 0.0
                accel_y = (state_n['vel_y_world'] - prev_state_n_minus_1['vel_y_world']) / dt_actual if dt_actual > 0 else 0.0
                logger.info(f"[検証3] 加速度（n-1→n）:")
                logger.info(f"  accel_x={accel_x*100:.2f} accel_y={accel_y*100:.2f} cm/s²")
                
                # 加速度を考慮した位置変化の予測（等加速度運動を仮定）
                # 位置変化 = 初速度×時間 + 0.5×加速度×時間²
                expected_dx_with_accel = prev_state_n_minus_1['vel_x_world'] * dt_actual + 0.5 * accel_x * dt_actual**2
                expected_dy_with_accel = prev_state_n_minus_1['vel_y_world'] * dt_actual + 0.5 * accel_y * dt_actual**2
                logger.info(f"[検証4] 加速度を考慮した位置変化の予測:")
                logger.info(f"  期待されるΔx={expected_dx_with_accel*100:.2f} cm 実際のΔx={dx*100:.2f} cm 差={abs(dx - expected_dx_with_accel)*100:.4f} cm")
                logger.info(f"  期待されるΔy={expected_dy_with_accel*100:.2f} cm 実際のΔy={dy*100:.2f} cm 差={abs(dy - expected_dy_with_accel)*100:.4f} cm")
                
                # 速度の積分検証（台形公式）
                expected_dx_trapezoidal = vel_x_avg * dt_actual
                expected_dy_trapezoidal = vel_y_avg * dt_actual
                logger.info(f"[検証5] 速度の積分（台形公式）による位置変化の予測:")
                logger.info(f"  期待されるΔx={expected_dx_trapezoidal*100:.2f} cm 実際のΔx={dx*100:.2f} cm 差={abs(dx - expected_dx_trapezoidal)*100:.4f} cm")
                logger.info(f"  期待されるΔy={expected_dy_trapezoidal*100:.2f} cm 実際のΔy={dy*100:.2f} cm 差={abs(dy - expected_dy_trapezoidal)*100:.4f} cm")
                
                # 機体座標系での検証
                vel_x_body_avg = (prev_state_n_minus_1['vel_x_body'] + state_n['vel_x_body']) / 2.0
                vel_y_body_avg = (prev_state_n_minus_1['vel_y_body'] + state_n['vel_y_body']) / 2.0
                
                # 方法1: nのyaw角度のみを使用（従来の方法）
                dx_body_simple = dx * cos_yaw_n + dy * sin_yaw_n
                dy_body_simple = -dx * sin_yaw_n + dy * cos_yaw_n
                vel_x_body_from_pos_simple = dx_body_simple / dt_actual if dt_actual > 0 else 0.0
                vel_y_body_from_pos_simple = dy_body_simple / dt_actual if dt_actual > 0 else 0.0
                
                # 方法2: 各ステップごとに座標変換を行い、合計を計算（高精度）
                dx_body_accurate = 0.0
                dy_body_accurate = 0.0
                if len(step_positions) > 0:
                    for step_dx, step_dy, step_yaw in step_positions:
                        cos_step_yaw = np.cos(step_yaw)
                        sin_step_yaw = np.sin(step_yaw)
                        dx_body_accurate += step_dx * cos_step_yaw + step_dy * sin_step_yaw
                        dy_body_accurate += -step_dx * sin_step_yaw + step_dy * cos_step_yaw
                vel_x_body_from_pos_accurate = dx_body_accurate / dt_actual if dt_actual > 0 else 0.0
                vel_y_body_from_pos_accurate = dy_body_accurate / dt_actual if dt_actual > 0 else 0.0
                
                logger.info(f"[検証6] 機体座標系での速度検証:")
                logger.info(f"  方法1（nのyaw角度のみ）: vel_x_body={vel_x_body_from_pos_simple*100:.2f} vel_y_body={vel_y_body_from_pos_simple*100:.2f} cm/s")
                logger.info(f"  方法2（各ステップごとの変換）: vel_x_body={vel_x_body_from_pos_accurate*100:.2f} vel_y_body={vel_y_body_from_pos_accurate*100:.2f} cm/s")
                logger.info(f"  速度の平均（台形公式）: vel_x_body_avg={vel_x_body_avg*100:.2f} vel_y_body_avg={vel_y_body_avg*100:.2f} cm/s")
                logger.info(f"  方法1との差: Δvel_x_body={(vel_x_body_from_pos_simple - vel_x_body_avg)*100:.2f} Δvel_y_body={(vel_y_body_from_pos_simple - vel_y_body_avg)*100:.2f} cm/s")
                logger.info(f"  方法2との差: Δvel_x_body={(vel_x_body_from_pos_accurate - vel_x_body_avg)*100:.2f} Δvel_y_body={(vel_y_body_from_pos_accurate - vel_y_body_avg)*100:.2f} cm/s")
                
                # 比較用に、方法2の結果を使用（より正確）
                vel_x_body_from_pos = vel_x_body_from_pos_accurate
                vel_y_body_from_pos = vel_y_body_from_pos_accurate
                
                # 比較と評価
                logger.info(f"[比較] 速度の一致度（ワールド座標系）:")
                diff_x = abs(vel_x_from_pos - vel_x_avg) * 100
                diff_y = abs(vel_y_from_pos - vel_y_avg) * 100
                logger.info(f"  位置変化から計算 vs 速度平均: Δvel_x={diff_x:.4f} Δvel_y={diff_y:.4f} cm/s")
                
                # 評価基準（誤差が小さいほど良い）
                tolerance = 0.1  # 0.1 cm/s の許容誤差
                if diff_x < tolerance and diff_y < tolerance:
                    logger.info(f"  ✅ 速度の一致度: 良好（誤差 < {tolerance} cm/s）")
                elif diff_x < tolerance * 10 and diff_y < tolerance * 10:
                    logger.info(f"  ⚠️  速度の一致度: やや悪い（誤差 < {tolerance * 10} cm/s）")
                else:
                    logger.info(f"  ❌ 速度の一致度: 悪い（誤差 >= {tolerance * 10} cm/s）")
                
                # 位置変化の予測精度
                logger.info(f"[比較] 位置変化の予測精度:")
                pos_error_x = abs(dx - expected_dx_trapezoidal) * 100
                pos_error_y = abs(dy - expected_dy_trapezoidal) * 100
                logger.info(f"  台形公式による予測誤差: Δx誤差={pos_error_x:.4f} cm Δy誤差={pos_error_y:.4f} cm")
                pos_error_x_accel = abs(dx - expected_dx_with_accel) * 100
                pos_error_y_accel = abs(dy - expected_dy_with_accel) * 100
                logger.info(f"  加速度考慮による予測誤差: Δx誤差={pos_error_x_accel:.4f} cm Δy誤差={pos_error_y_accel:.4f} cm")
                
                pos_tolerance = 0.01  # 0.01 cm の許容誤差
                if pos_error_x < pos_tolerance and pos_error_y < pos_tolerance:
                    logger.info(f"  ✅ 位置変化の予測精度: 良好（誤差 < {pos_tolerance} cm）")
                elif pos_error_x < pos_tolerance * 10 and pos_error_y < pos_tolerance * 10:
                    logger.info(f"  ⚠️  位置変化の予測精度: やや悪い（誤差 < {pos_tolerance * 10} cm）")
                else:
                    logger.info(f"  ❌ 位置変化の予測精度: 悪い（誤差 >= {pos_tolerance * 10} cm）")
                
                # Simpson法による検証（3点が必要、n-2の状態が存在する場合のみ）
                if prev_state_n_minus_2 is not None and prev_t_n_minus_2 is not None and prev_t_n_minus_1 is not None:
                    dt_prev = prev_t_n_minus_1 - prev_t_n_minus_2
                    # ログ間隔のチェックを緩和（1%の誤差まで許容）
                    dt_tolerance = max(dt_actual, dt_prev) * 0.01  # 1%の誤差まで許容
                    is_equal_interval = abs(dt_prev - dt_actual) < dt_tolerance
                    
                    if is_equal_interval:
                        # Simpson法: Δpos = (dt/6) * [vel_n-2 + 4*vel_n-1 + vel_n]
                        # 注意: dt_actualを使用（n-1からnへの時間間隔）
                        expected_dx_simpson = (dt_actual / 6.0) * (
                            prev_state_n_minus_2['vel_x_world'] + 
                            4.0 * prev_state_n_minus_1['vel_x_world'] + 
                            state_n['vel_x_world']
                        )
                        expected_dy_simpson = (dt_actual / 6.0) * (
                            prev_state_n_minus_2['vel_y_world'] + 
                            4.0 * prev_state_n_minus_1['vel_y_world'] + 
                            state_n['vel_y_world']
                        )
                        logger.info(f"[検証7] Simpson法による位置変化の予測（高精度、等間隔）:")
                        logger.info(f"  時間間隔: dt_prev={dt_prev:.4f}s dt_actual={dt_actual:.4f}s 差={abs(dt_prev - dt_actual):.6f}s")
                        logger.info(f"  期待されるΔx={expected_dx_simpson*100:.2f} cm 実際のΔx={dx*100:.2f} cm 差={abs(dx - expected_dx_simpson)*100:.4f} cm")
                        logger.info(f"  期待されるΔy={expected_dy_simpson*100:.2f} cm 実際のΔy={dy*100:.2f} cm 差={abs(dy - expected_dy_simpson)*100:.4f} cm")
                        
                        simpson_error_x = abs(dx - expected_dx_simpson) * 100
                        simpson_error_y = abs(dy - expected_dy_simpson) * 100
                        if simpson_error_x < pos_tolerance and simpson_error_y < pos_tolerance:
                            logger.info(f"  ✅ Simpson法の予測精度: 良好（誤差 < {pos_tolerance} cm）")
                        elif simpson_error_x < pos_tolerance * 10 and simpson_error_y < pos_tolerance * 10:
                            logger.info(f"  ⚠️  Simpson法の予測精度: やや悪い（誤差 < {pos_tolerance * 10} cm）")
                        else:
                            logger.info(f"  ❌ Simpson法の予測精度: 悪い（誤差 >= {pos_tolerance * 10} cm）")
                    else:
                        # 等間隔でない場合でも、Simpson法を適用（ただし警告を出す）
                        logger.info(f"[検証7] Simpson法による位置変化の予測（非等間隔、参考値）:")
                        logger.info(f"  ⚠️  警告: ログ間隔が等間隔でない（dt_prev={dt_prev:.4f}s dt_actual={dt_actual:.4f}s 差={abs(dt_prev - dt_actual):.6f}s）")
                        # 平均時間間隔を使用
                        dt_avg = (dt_prev + dt_actual) / 2.0
                        expected_dx_simpson = (dt_avg / 6.0) * (
                            prev_state_n_minus_2['vel_x_world'] + 
                            4.0 * prev_state_n_minus_1['vel_x_world'] + 
                            state_n['vel_x_world']
                        )
                        expected_dy_simpson = (dt_avg / 6.0) * (
                            prev_state_n_minus_2['vel_y_world'] + 
                            4.0 * prev_state_n_minus_1['vel_y_world'] + 
                            state_n['vel_y_world']
                        )
                        logger.info(f"  期待されるΔx={expected_dx_simpson*100:.2f} cm 実際のΔx={dx*100:.2f} cm 差={abs(dx - expected_dx_simpson)*100:.4f} cm")
                        logger.info(f"  期待されるΔy={expected_dy_simpson*100:.2f} cm 実際のΔy={dy*100:.2f} cm 差={abs(dy - expected_dy_simpson)*100:.4f} cm")
                        logger.info(f"  ⚠️  注意: 非等間隔のため、Simpson法の精度は低下する可能性があります")
                else:
                    # n-2の状態が存在しない場合
                    if prev_state_n_minus_2 is None:
                        logger.info(f"[検証7] Simpson法による位置変化の予測:")
                        logger.info(f"  ⚠️  スキップ: n-2の状態が存在しません（初回ログ出力の可能性）")
                    elif prev_t_n_minus_2 is None:
                        logger.info(f"[検証7] Simpson法による位置変化の予測:")
                        logger.info(f"  ⚠️  スキップ: n-2の時刻が存在しません")
                
                # ピッチ前傾テスト中のマーカー
                if is_pitch_test:
                    logger.info(f"  [ピッチ前傾テスト中]")
                
                logger.info("")
                
                # 前回の値を更新（位置変化計算用）
                prev_x = x
                prev_y = y
                prev_t = t
                
                # ログ間隔の間の記録をクリア（次のログ間隔の記録を開始）
                step_positions = []
                step_positions_prev_x = x
                step_positions_prev_y = y
                
                # Simpson法の検証用に、n-2の状態を更新（ログ出力時点での時刻を保存）
                # 注意: ログ間隔が固定（10ステップ）なので、等間隔が保証される
                # ログ出力時に、前回のログ出力時の状態をn-2として保存
                # 現在のログ出力時の状態をn-1として保存（次回のログ出力時にn-2として使用）
                if prev_t_n_minus_1 is not None:
                    # 前回のログ出力時の状態をn-2として保存
                    prev_t_n_minus_2 = prev_t_n_minus_1
                    prev_state_n_minus_2 = prev_state_n_minus_1.copy() if prev_state_n_minus_1 is not None else None
                # 現在のログ出力時の状態をn-1として保存
                prev_t_n_minus_1 = t
                prev_state_n_minus_1 = state_n.copy()
            
            # コンソールには1秒ごと（240ステップごと）に簡易情報を表示
            if step % 240 == 0:
                r, pitch, y = state.get('roll', 0), state.get('pitch', 0), state.get('yaw', 0)
                print(f"t={t:.1f}s  x={x*100:.1f} y={y*100:.1f} z={z:.2f}  r={np.degrees(r):.1f}° p={np.degrees(pitch):.1f}° y={np.degrees(y):.1f}°")
            
            time.sleep(dt)
    
    except KeyboardInterrupt:
        print("\n⏸️ 中断")
        logger.info("シミュレーション中断")
    
    p.disconnect()
    print("✅ シミュレーション終了")
    logger.info("シミュレーション終了")


if __name__ == "__main__":
    main()
