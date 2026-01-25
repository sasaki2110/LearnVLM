"""
forward_quadrotor: チョン応答検証（目標位置なし・開放系）

【1 チョン・長さスイープ】1) ホバ 2) チョン（CHON_DURATION 可変） 3) ホバ
→ チョン／チョーン／チョーーン／チョーーーン と長さを変え、「長さ→Δx」の感度を取る。
  スイープ: CHON_DURATION = 0.2, 0.3, 0.4, 0.5 [s] で各 1 run。
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import logging
from datetime import datetime

# ============================================================================
# PID制御器（pid_quadrotor と同じ）
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

    def compute(self, target, current, current_vel, dt, freeze_integral=False):
        error = target - current
        if not freeze_integral:
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        output = self.kp * error + self.ki * self.integral + self.kd * (-current_vel)
        self.last_error = error
        return output

    def reset(self):
        self.integral = 0.0


# ============================================================================
# Quadrotor制御（pid と同じ + チョン用の roll 上書きだけ）
# ============================================================================

class QuadrotorController:
    """pid_quadrotor と同一。chon_roll_override がセットされている間は roll 目標をそれで上書き。
    位置・速度の外側PDを追加可能（慣性ドリフト低減用）。
    roll入力時のyaw補正機能を追加（ドリフト低減用）。
    """

    def __init__(self, robot_id, enable_position_velocity_pd=False,
                 kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0,
                 target_x=None, target_y=None,
                 enable_yaw_compensation=False, yaw_comp_gain=0.0):
        """
        Args:
            robot_id: PyBulletのロボットID
            enable_position_velocity_pd: 位置・速度PDを有効にするか
            kp_x, kd_x: X方向の位置P・速度Dゲイン
            kp_y, kd_y: Y方向の位置P・速度Dゲイン
            target_x, target_y: 目標位置（Noneの場合は位置Pを無効化、Dのみ使用）
            enable_yaw_compensation: roll入力時のyaw補正を有効にするか
            yaw_comp_gain: yaw補正ゲイン（roll変化率に対するyawトルク補正の比率）
        """
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
        # roll/pitch: kd（D=ショックアブソーバー）を強くしてチョン後の振動・ブレを抑制
        # kpを上げすぎると目標姿勢の変化が大きい場合に追従できず不安定になるため、適度な値に設定
        # roll/pitch PID: kpを下げ、kdを上げて振動を抑制
        self.roll_pid = PIDController(kp=5.0, ki=0.2, kd=15.0)   # kp: 10→5, ki: 0.5→0.2, kd: 10→15
        self.pitch_pid = PIDController(kp=5.0, ki=0.2, kd=15.0)  # kp: 10→5, ki: 0.5→0.2, kd: 10→15
        self.yaw_pid = PIDController(kp=10.0, ki=0.3, kd=5.0)  # yaw回転を抑制するため強化

        self.target_height = 2.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.chon_roll_override = None  # チョン時: この値 [rad] を roll 目標にする。None なら target_roll

        # 位置・速度PDのパラメータ
        self.enable_position_velocity_pd = enable_position_velocity_pd
        self.kp_x = kp_x
        self.kd_x = kd_x
        self.kp_y = kp_y
        self.kd_y = kd_y
        # target_x/yがNoneの場合は位置Pを無効化（Dのみ使用）
        self.target_x = target_x
        self.target_y = target_y

        # roll入力時のyaw補正パラメータ
        # 原理: roll変化率（thrust_roll）に比例してyawトルクを補正
        # prop1とprop3は同じ回転方向（時計回り）→ rollを変化させると反トルクが偏りyaw回転が発生
        # この反トルクを打ち消すために、roll変化率に比例したyawトルクを追加
        self.enable_yaw_compensation = enable_yaw_compensation
        self.yaw_comp_gain = yaw_comp_gain
        self.last_thrust_roll = 0.0  # 前回のthrust_roll（変化率計算用）

        self.mass = 0.5
        self.gravity = 9.81
        self.base_thrust = self.mass * self.gravity / 4.0
        self.max_thrust = 20.0
        self.min_thrust = 0.0

    def update(self, dt):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        vel_z = vel[2]
        roll_vel, pitch_vel, yaw_vel = ang_vel[0], ang_vel[1], ang_vel[2]
        
        # 機体座標系への速度変換（yaw角度を使用）
        # ワールド座標系から機体座標系への変換
        # URDF定義: prop1=+X（右）, prop2=+Y（前）
        # 機体のX軸：右方向（prop1が+X）
        # 機体のY軸：前方向（prop2が+Y）
        # 
        # 変換行列（ワールド→機体）:
        # [vel_x_body]   [cos(yaw)  sin(yaw)] [vel_x_world]
        # [vel_y_body] = [-sin(yaw) cos(yaw)] [vel_y_world]
        #
        # 逆変換（機体→ワールド）:
        # [vel_x_world]   [cos(yaw) -sin(yaw)] [vel_x_body]
        # [vel_y_world] = [sin(yaw)  cos(yaw)] [vel_y_body]
        #
        vel_x_world = vel[0]  # ワールド座標系のX方向速度（変換用のみ）
        vel_y_world = vel[1]  # ワールド座標系のY方向速度（変換用のみ）
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vel_x_body = vel_x_world * cos_yaw + vel_y_world * sin_yaw  # 機体のX方向（右方向、prop1方向）
        vel_y_body = -vel_x_world * sin_yaw + vel_y_world * cos_yaw  # 機体のY方向（前方向、prop2方向）

        # 基本のroll目標（chon_roll_overrideがあればそれ、なければtarget_roll）
        use_roll = self.chon_roll_override if self.chon_roll_override is not None else self.target_roll
        use_pitch = self.target_pitch

        # 位置・速度PDによるオフセット（デバッグ用に初期化）
        roll_offset = 0.0
        pitch_offset = 0.0

            # 位置・速度PDを外側ループとして追加（roll/pitchの目標値に加算）
        if self.enable_position_velocity_pd:
            x, y = pos[0], pos[1]
            # 位置P: 位置誤差を機体座標系に変換してから、roll/pitchの目標値を計算
            # 速度D: roll += -kd_x * vel_x_body, pitch += -kd_y * vel_y_body（機体座標系の速度を使用）
            # 符号: 正のvel_x_body（機体の+X方向の速度）→ 負のroll（-roll）で減速
            #       正のvel_y_body（機体の+Y方向の速度）→ 負のpitch（-pitch）で減速
            # 注意: roll/pitchの制御は機体座標系で動作するため、位置Pと速度Dの両方を機体座標系で計算する必要がある
            
            # 位置P（target_x/yがNoneでない場合のみ、かつチョン中は無効化）
            # チョン中（chon_roll_overrideが有効）は位置Pを無効化し、
            # チョン終了後（chon_roll_overrideがNone）のみ位置Pを有効化
            # 修正: 位置誤差をワールド座標系から機体座標系に変換してから使用
            if self.chon_roll_override is None:
                if self.target_x is not None or self.target_y is not None:
                    # ワールド座標系の位置誤差を計算
                    dx_world = (self.target_x - x) if self.target_x is not None else 0.0
                    dy_world = (self.target_y - y) if self.target_y is not None else 0.0
                    
                    # 機体座標系に変換（速度と同じ変換式を使用）
                    # 機体のX軸：右方向（prop1方向）
                    # 機体のY軸：前方向（prop2方向）
                    dx_body = dx_world * cos_yaw + dy_world * sin_yaw  # 機体のX方向（右方向）
                    dy_body = -dx_world * sin_yaw + dy_world * cos_yaw  # 機体のY方向（前方向）
                    
                    # 機体座標系の位置誤差を使ってroll/pitchの目標値を計算
                    if self.target_x is not None:
                        # 機体が+X方向（右）にずれている → 正のrollで-X方向（左）に戻す
                        roll_offset += self.kp_x * dx_body
                    if self.target_y is not None:
                        # 機体が+Y方向（前）にずれている → 負のpitchで-Y方向（後）に戻す
                        # 符号: 正のdy_body（機体の+Y方向の位置誤差）→ 負のpitch（-pitch）で戻す
                        pitch_offset += -self.kp_y * dy_body
            
            # 速度D
            # 修正: ワールド座標系の速度（vel_x, vel_y）ではなく、機体座標系の速度（vel_x_body, vel_y_body）を使用
            # X方向: -kd_x * vel_x_body（機体の+X方向の速度 → rollで減速）
            # Y方向: -kd_y * vel_y_body（機体の+Y方向の速度 → pitchで減速）
            # チョン中: Y方向のみ速度Dを有効化（X方向はチョンによる加速を妨げないように無効化）
            if self.chon_roll_override is None:
                roll_offset += -self.kd_x * vel_x_body
            pitch_offset += -self.kd_y * vel_y_body  # Y方向は常に有効（ドリフト抑制）
            
            # 解決策1: 位置・速度PDの出力を姿勢PIDの目標値に加算しない（姿勢PIDを経由しない）
            # 代わりに、roll_offsetとpitch_offsetを直接推力配分に反映する
            # use_roll += roll_offset  # 削除: 姿勢PIDを経由しない
            # use_pitch += pitch_offset  # 削除: 姿勢PIDを経由しない

        # チョン中は height の integral を積み増ししない（吹き上がり防止）
        height_output = self.height_pid.compute(
            self.target_height, pos[2], vel_z, dt,
            freeze_integral=(self.chon_roll_override is not None),
        )
        roll_output = self.roll_pid.compute(use_roll, roll, roll_vel, dt)
        pitch_output = self.pitch_pid.compute(use_pitch, pitch, pitch_vel, dt)
        yaw_output = self.yaw_pid.compute(self.target_yaw, yaw, yaw_vel, dt)

        thrust_height_raw = self.base_thrust + height_output / 4.0
        # 姿勢PIDの出力を推力に変換
        thrust_roll_from_pid = roll_output / (2.0 * self.arm_length)
        thrust_pitch_from_pid = pitch_output / (2.0 * self.arm_length)
        
        # 解決策1: 位置・速度PDの出力（roll_offset, pitch_offset）を直接推力に変換して加算
        # 角度から推力への変換係数: 1 / (2.0 * arm_length)
        if self.enable_position_velocity_pd:
            thrust_roll_offset = roll_offset / (2.0 * self.arm_length)
            thrust_pitch_offset = pitch_offset / (2.0 * self.arm_length)
            thrust_roll = thrust_roll_from_pid + thrust_roll_offset
            thrust_pitch = thrust_pitch_from_pid + thrust_pitch_offset
        else:
            thrust_roll = thrust_roll_from_pid
            thrust_pitch = thrust_pitch_from_pid
        
        thrust_yaw = yaw_output * 0.1
        
        # roll入力時のyaw補正（フィードフォワード）
        # 原理: roll姿勢を維持するために対角ローター（prop1とprop3）の推力差が発生
        #       prop1とprop3は同じ回転方向（時計回り）→ 推力差が反トルクの偏りを生じyaw回転が発生
        # 
        # 補正方針: 実際のthrust_roll（姿勢PIDの出力 + 位置・速度PDの出力）に比例してyaw補正
        #           thrust_roll > 0 → prop1が増、prop3が減 → 反トルクが正方向に偏り → 機体は負のyaw方向に回転
        #           これを打ち消すために、正のyawトルクを追加
        yaw_compensation = 0.0
        if self.enable_yaw_compensation:
            # thrust_rollに比例してyaw補正
            # ローターのモーメント係数（prop_moments）を考慮
            # prop1: +0.0245, prop3: +0.0245（同じ回転方向）
            # thrust_roll > 0 → prop1 + thrust_roll, prop3 - thrust_roll
            # 反トルク変化 = thrust_roll * 0.0245 - (-thrust_roll) * 0.0245 = 2 * thrust_roll * 0.0245
            # これを打ち消すyawトルクが必要
            moment_coeff = abs(self.prop_moments[0])  # 0.0245
            yaw_compensation = self.yaw_comp_gain * thrust_roll * 2.0 * moment_coeff
            thrust_yaw += yaw_compensation
        
        # 前回のthrust_rollを保存（将来の変化率ベースの補正用）
        self.last_thrust_roll = thrust_roll

        # 推力の「負→0 クリップ」で roll/pitch が相殺されず見かけの合計が正になり
        # 吹き上がるのを防ぐ: (1) thrust_height<0 なら全 0 で落下 (2) それ以外は
        # roll/pitch をスケールして「どれかが 0 にクリップされる」状況を避ける。
        if thrust_height_raw < 0:
            thrust_height = 0.0
            thrust_roll = 0.0
            thrust_pitch = 0.0
            thrust_yaw = 0.0
        else:
            thrust_height = thrust_height_raw
            m = max(abs(thrust_roll), abs(thrust_pitch), 1e-9)
            if thrust_height < m:
                s = thrust_height / m
                thrust_roll *= s
                thrust_pitch *= s

        thrusts = [
            thrust_height + thrust_roll + thrust_yaw * self.prop_moments[0],
            thrust_height + thrust_pitch + thrust_yaw * self.prop_moments[1],
            thrust_height - thrust_roll + thrust_yaw * self.prop_moments[2],
            thrust_height - thrust_pitch + thrust_yaw * self.prop_moments[3],
        ]
        thrusts = [np.clip(t, self.min_thrust, self.max_thrust) for t in thrusts]

        return thrusts, {
            'height': pos[2],
            'x': pos[0],
            'y': pos[1],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'vel_x_body': vel_x_body,  # 機体座標系のX方向速度（制御に使用）
            'vel_y_body': vel_y_body,  # 機体座標系のY方向速度（制御に使用）
            'vel_x_world': vel_x_world,  # ワールド座標系のX方向速度（デバッグ用）
            'vel_y_world': vel_y_world,  # ワールド座標系のY方向速度（デバッグ用）
            'target_roll': use_roll,
            'target_pitch': use_pitch,
            'roll_offset': roll_offset if self.enable_position_velocity_pd else 0.0,  # 位置・速度PDによるrollオフセット（角度、デバッグ用）
            'pitch_offset': pitch_offset if self.enable_position_velocity_pd else 0.0,  # 位置・速度PDによるpitchオフセット（角度、デバッグ用）
            'thrust_roll_offset': (roll_offset / (2.0 * self.arm_length)) if self.enable_position_velocity_pd else 0.0,  # 位置・速度PDによるrollオフセット（推力、デバッグ用）
            'thrust_pitch_offset': (pitch_offset / (2.0 * self.arm_length)) if self.enable_position_velocity_pd else 0.0,  # 位置・速度PDによるpitchオフセット（推力、デバッグ用）
            'height_integral': self.height_pid.integral,
            'height_output': height_output,
            'roll_output': roll_output,  # 姿勢PIDのroll出力（デバッグ用）
            'pitch_output': pitch_output,  # 姿勢PIDのpitch出力（デバッグ用）
            'thrusts': thrusts.copy(),  # 各プロペラの推力（デバッグ用）
            'thrust_roll': thrust_roll,  # roll制御による推力差（デバッグ用）
            'thrust_pitch': thrust_pitch,  # pitch制御による推力差（デバッグ用）
            'thrust_yaw': thrust_yaw,  # yaw制御による推力（補正込み、デバッグ用）
            'yaw_compensation': yaw_compensation if self.enable_yaw_compensation else 0.0,  # yaw補正値（デバッグ用）
            'thrust_height': thrust_height,  # 高さ制御による基本推力（デバッグ用）
            'total_force_y': getattr(self, '_last_total_force_y', 0.0),  # 総推力のY方向への分解（デバッグ用）
            'prop_forces': getattr(self, '_last_prop_forces', []),  # 各プロペラの推力ベクトル（デバッグ用）
        }

    def apply_thrusts(self, thrusts):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        # デバッグ用: 推力の合計とY方向への分解を計算
        total_force_world = np.array([0.0, 0.0, 0.0])
        # デバッグ用: 各プロペラの推力ベクトルを保存
        prop_forces = []
        for i, (thrust, prop_pos_local) in enumerate(zip(thrusts, self.prop_positions)):
            force_local = [0, 0, thrust]
            force_world = p.rotateVector(orn, force_local)
            if isinstance(force_world, tuple):
                force_world = list(force_world)
            force_world_array = np.array(force_world)
            total_force_world += force_world_array
            prop_forces.append({
                'prop_id': i,
                'thrust': thrust,
                'force_world': force_world_array.copy(),
                'prop_pos_local': prop_pos_local.copy() if isinstance(prop_pos_local, np.ndarray) else np.array(prop_pos_local)
            })
            pl = list(prop_pos_local) if isinstance(prop_pos_local, np.ndarray) else prop_pos_local
            prop_world, _ = p.multiplyTransforms(pos, orn, pl, [0, 0, 0, 1])
            if isinstance(prop_world, tuple):
                prop_world = list(prop_world)
            p.applyExternalForce(self.robot_id, -1, force_world, prop_world, p.WORLD_FRAME)
        # デバッグ用: 総推力のY方向への分解を保存
        self._last_total_force_y = total_force_world[1]
        self._last_prop_forces = prop_forces

    def reset(self):
        # height_pid はリセットしない（チョン後の吹き上がり防止。integral をゼロにすると
        # 過渡応答が大きくなり、高度が暴れることがある）
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()


# ============================================================================
# メイン: ホバ → チョン1（固定 roll × 固定時間）→ ホバ → 位置変化を報告
# ============================================================================

def main():
    print("=" * 60)
    print("forward_quadrotor: チョン1の応答検証（目標位置なし）")
    print("=" * 60)

    device_id = p.connect(p.GUI)
    if device_id < 0:
        print("❌ PyBullet接続失敗")
        exit(1)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    quadrotor_dir = os.path.join(project_root, "Quadrotor")
    data_path = pybullet_data.getDataPath()
    p.setAdditionalSearchPath(data_path)
    p.setAdditionalSearchPath(quadrotor_dir)
    p.setGravity(0, 0, -9.81)

    plane_path = os.path.join(data_path, "plane.urdf")
    p.loadURDF(plane_path)
    quadrotor_path = os.path.join(quadrotor_dir, "quadrotor.urdf")
    robot_id = p.loadURDF(quadrotor_path, [0, 0, 1.0])

    # 位置・速度PDの設定（課題2: 慣性ドリフト低減）
    ENABLE_POSITION_VELOCITY_PD = True  # True にすると位置・速度PDが有効
    KP_X = 0.050  # X方向の位置Pゲイン（Run 26の設定に戻す）
    KD_X = 0.300  # X方向の速度Dゲイン（Run 26の設定に戻す）
    KP_Y = 0.050  # Y方向の位置Pゲイン（Run 26の設定に戻す）
    KD_Y = 0.5  # Y方向の速度Dゲイン（調整：ドリフト抑制と安定性のバランス）
    TARGET_X = 0.0  # 目標X位置（原点に戻す）
    TARGET_Y = 0.0  # 目標Y位置（原点に戻す）
    
    # roll入力時のyaw補正の設定（課題: yaw回転によるドリフト低減）
    # 原理: roll変化（thrust_roll）によって対角ローター（prop1とprop3）の推力が変化
    #       prop1とprop3は同じ回転方向（時計回り）→ 推力が変わると反トルクが偏りyaw回転が発生
    # 補正: thrust_rollに比例してyawトルクを追加し、反トルクの偏りを打ち消す
    ENABLE_YAW_COMPENSATION = True  # True にするとyaw補正が有効
    YAW_COMP_GAIN = -0.3  # yaw補正ゲイン（Δx/Δyのバランスを考慮した推奨値）
    # ゲインの調整指針:
    # - 小さすぎる → yaw回転が残る
    # - 大きすぎる → 逆方向にyaw回転する
    # - 適切な値 → yaw回転がほぼゼロになる

    controller = QuadrotorController(
        robot_id,
        enable_position_velocity_pd=ENABLE_POSITION_VELOCITY_PD,
        kp_x=KP_X, kd_x=KD_X,
        kp_y=KP_Y, kd_y=KD_Y,
        target_x=TARGET_X, target_y=TARGET_Y,
        enable_yaw_compensation=ENABLE_YAW_COMPENSATION,
        yaw_comp_gain=YAW_COMP_GAIN
    )
    controller.target_height = 2.0
    controller.target_roll = 0.0
    controller.target_pitch = 0.0
    controller.target_yaw = 0.0
    controller.chon_roll_override = None

    dt = 1.0 / 240.0
    step = 0
    phase = "hover_start"
    phase_start_step = 0
    T_HOVER = 3.0
    T_HOVER_MEASURE = 2.5

    # 1 チョンの長さスイープ: チョン 0.2 → チョーン 0.3 → チョーーン 0.4 → チョーーーン 0.5
    # 各 run で CHON_DURATION だけ変えて「長さ→Δx」の感度を取る。
    CHON_ROLL = 0.03   # rad ≒ 1.72°（最大値）
    CHON_DURATION = 5.0   # [s] スイープ: 0.2, 0.3, 0.4, 0.5
    
    # チョン後半で徐々にrollを減らす（慣性低減のため）
    ENABLE_GRADUAL_REDUCTION = True  # 徐々に減らす機能を有効にするか
    GRADUAL_REDUCTION_START = 1.0  # 減衰開始時刻 [s]（チョン開始から、2.0 → 1.0に変更）
    GRADUAL_REDUCTION_DURATION = 4.0  # 減衰期間 [s]（1.0秒から5.0秒まで、3.0 → 4.0に変更）

    x_start, y_start, z_start = 0.0, 0.0, 0.0

    # ロギング設定（p00_sampleのロギング設定を参考）
    log_dir = os.path.join(project_root, "samples", "quadrotor", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "forward_quadrotor_velocity.log")
    log_file_rotated = os.path.join(log_dir, "forward_quadrotor_velocity.log.1")
    
    # ログローテーション: 既存のログファイルがあれば.1にリネーム
    if os.path.exists(log_file):
        if os.path.exists(log_file_rotated):
            os.remove(log_file_rotated)  # 古い.1ファイルを削除
        os.rename(log_file, log_file_rotated)
    
    # ロガーを設定
    logger = logging.getLogger('forward_quadrotor')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 既存のハンドラーをクリア
    
    # ファイルハンドラーを追加（新規ファイルとして作成）
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラーも追加（INFOレベル以上のみ）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # コンソールにはWARNING以上のみ
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    logger.info("=" * 60)
    logger.info("forward_quadrotor: チョン1の応答検証（目標位置なし）")
    logger.info("=" * 60)
    logger.info(f"ログファイル: {log_file}")
    
    chon_desc = f"roll={np.degrees(CHON_ROLL):.3f}° × {CHON_DURATION}s"
    if ENABLE_GRADUAL_REDUCTION:
        chon_desc += f" (減衰: {GRADUAL_REDUCTION_START}sから{GRADUAL_REDUCTION_DURATION}sで0まで)"
    print(f"\n📌 1) ホバ {T_HOVER}s  2) チョン: {chon_desc}  3) ホバ {T_HOVER_MEASURE}s → 位置変化を出力")
    logger.info(f"📌 1) ホバ {T_HOVER}s  2) チョン: {chon_desc}  3) ホバ {T_HOVER_MEASURE}s → 位置変化を出力")
    if ENABLE_POSITION_VELOCITY_PD:
        print(f"   位置・速度PD: 有効  kp_x={KP_X:.3f} kd_x={KD_X:.3f} kp_y={KP_Y:.3f} kd_y={KD_Y:.3f}")
        logger.info(f"   位置・速度PD: 有効  kp_x={KP_X:.3f} kd_x={KD_X:.3f} kp_y={KP_Y:.3f} kd_y={KD_Y:.3f}")
        if TARGET_X is not None or TARGET_Y is not None:
            print(f"   目標位置: x={TARGET_X}, y={TARGET_Y}")
            logger.info(f"   目標位置: x={TARGET_X}, y={TARGET_Y}")
        else:
            print(f"   目標位置: なし（Dのみ使用）")
            logger.info(f"   目標位置: なし（Dのみ使用）")
    else:
        print(f"   位置・速度PD: 無効（開放系チョン実験）")
        logger.info(f"   位置・速度PD: 無効（開放系チョン実験）")
    
    if ENABLE_YAW_COMPENSATION:
        print(f"   yaw補正: 有効  gain={YAW_COMP_GAIN:.3f}")
        logger.info(f"   yaw補正: 有効  gain={YAW_COMP_GAIN:.3f}")
    else:
        print(f"   yaw補正: 無効")
        logger.info(f"   yaw補正: 無効")
    print(f"   ログファイル: {log_file}")
    print()

    try:
        while True:
            thrusts, state = controller.update(dt)
            controller.apply_thrusts(thrusts)
            p.stepSimulation()
            t = step * dt
            step += 1
            
            # 重要: stepSimulation()後の位置・姿勢を取得（valcheck.pyと同じ方式）
            # state['x'], state['y']はstepSimulation()前の値なので、位置変化の記録には使用しない
            pos_after, orn_after = p.getBasePositionAndOrientation(robot_id)
            euler_after = p.getEulerFromQuaternion(orn_after)
            x_after, y_after, z_after = pos_after[0], pos_after[1], pos_after[2]
            roll_after, pitch_after, yaw_after = euler_after
            
            # stateからの値（stepSimulation()前の値、表示用に残す）
            x, y, z = state['x'], state['y'], state['height']
            elapsed = (step - phase_start_step) * dt
            r, pitch, yaw = state.get('roll', 0), state.get('pitch', 0), state.get('yaw', 0)

            if phase == "hover_start":
                if t >= T_HOVER:
                    x_start, y_start, z_start = x_after, y_after, z_after
                    controller.chon_roll_override = CHON_ROLL
                    phase = "chon"
                    phase_start_step = step
                    print(f"  [chon] 開始 入力 roll={CHON_ROLL:.3f}rad 直前 (x,y,z)=({x_after*100:.2f},{y_after*100:.2f},{z_after:.2f}) (r,p,y)=({np.degrees(roll_after):.2f},{np.degrees(pitch_after):.2f},{np.degrees(yaw_after):.2f})°")
            elif phase == "chon":
                # チョン後半で徐々にrollを減らす（慣性低減のため）
                if ENABLE_GRADUAL_REDUCTION and elapsed >= GRADUAL_REDUCTION_START:
                    # 減衰開始時刻以降、線形に減らす
                    reduction_elapsed = elapsed - GRADUAL_REDUCTION_START
                    if reduction_elapsed >= GRADUAL_REDUCTION_DURATION:
                        # 減衰期間が終了したら、roll=0
                        controller.chon_roll_override = 0.0
                    else:
                        # 線形減衰: roll = CHON_ROLL * (1 - reduction_elapsed / GRADUAL_REDUCTION_DURATION)
                        reduction_ratio = 1.0 - (reduction_elapsed / GRADUAL_REDUCTION_DURATION)
                        controller.chon_roll_override = CHON_ROLL * reduction_ratio
                else:
                    # 減衰開始前は、最大値
                    controller.chon_roll_override = CHON_ROLL
                
                # 1秒ごとにログファイルに出力
                if (step - phase_start_step) % 240 == 0:
                    vel_x_body = state.get('vel_x_body', 0)
                    vel_y_body = state.get('vel_y_body', 0)
                    logger.info(f"[chon] t={t:.2f}s x={x_after*100:.2f} y={y_after*100:.2f} vel_x={vel_x_body*100:.2f} vel_y={vel_y_body*100:.2f} cm/s")
                
                if elapsed >= CHON_DURATION:
                    controller.chon_roll_override = None
                    phase = "hover_measure"
                    phase_start_step = step
                    # stepSimulation()後の位置・姿勢・速度を取得
                    vel_after, _ = p.getBaseVelocity(robot_id)
                    vel_x_body_after = vel_after[0] * np.cos(yaw_after) + vel_after[1] * np.sin(yaw_after)
                    vel_y_body_after = -vel_after[0] * np.sin(yaw_after) + vel_after[1] * np.cos(yaw_after)
                    print(f"  [chon] 終了 直後 (x,y,z)=({x_after*100:.2f},{y_after*100:.2f},{z_after:.2f}) (r,p,y)=({np.degrees(roll_after):.2f},{np.degrees(pitch_after):.2f},{np.degrees(yaw_after):.2f})° vel_body=(x:{vel_x_body_after*100:.2f},y:{vel_y_body_after*100:.2f})cm/s")
            elif phase == "hover_measure":
                # チョン終了直後〜0.5 s: 0.1 s ごとにデバッグ（高度・height PID の状態、目標姿勢と実際の姿勢）
                if elapsed <= 0.5 and (step - phase_start_step) % 24 == 0:
                    hi = state.get('height_integral', 0)
                    ho = state.get('height_output', 0)
                    r, pitch, y = state.get('roll', 0), state.get('pitch', 0), state.get('yaw', 0)
                    target_r = state.get('target_roll', 0)
                    target_p = state.get('target_pitch', 0)
                    print(f"  [debug] t={t:.2f}s z={z:.3f} target=2.0 hi={hi:.3f} ho={ho:.3f}")
                    print(f"          r={np.degrees(r):.2f}° (target={np.degrees(target_r):.2f}°) p={np.degrees(pitch):.2f}° (target={np.degrees(target_p):.2f}°) y={np.degrees(y):.2f}°")
                
                # 1秒ごとにログ出力（簡素化版）
                if (step - phase_start_step) % 240 == 0:
                    vel_x_body = state.get('vel_x_body', 0)
                    vel_y_body = state.get('vel_y_body', 0)
                    logger.info(f"[hover] t={t:.2f}s x={x_after*100:.2f} y={y_after*100:.2f} vel_x={vel_x_body*100:.2f} vel_y={vel_y_body*100:.2f} cm/s")

                if elapsed >= T_HOVER_MEASURE:
                    # 重要: stepSimulation()後の位置（x_after, y_after）を使用
                    dx, dy, dz = (x_after - x_start) * 100, (y_after - y_start) * 100, (z_after - z_start) * 100
                    # stepSimulation()後の速度を取得
                    vel_after_final, _ = p.getBaseVelocity(robot_id)
                    vel_x_body_final = vel_after_final[0] * np.cos(yaw_after) + vel_after_final[1] * np.sin(yaw_after)
                    vel_y_body_final = -vel_after_final[0] * np.sin(yaw_after) + vel_after_final[1] * np.cos(yaw_after)
                    print(f"\n--- チョン1回（長さ {CHON_DURATION}s）による位置変化（ホバ開始時 → ホバ計測後）---")
                    print(f"  Δx = {dx:+.2f} cm")
                    print(f"  Δy = {dy:+.2f} cm")
                    print(f"  Δz = {dz:+.2f} cm")
                    print(f"  最終位置: (x,y,z)=({x_after*100:.2f},{y_after*100:.2f},{z_after:.3f}) cm,cm,m")
                    print(f"  最終速度（機体座標系）: vel_x_body={vel_x_body_final*100:.2f} cm/s, vel_y_body={vel_y_body_final*100:.2f} cm/s")
                    print(f"  最終姿勢: (r,p,y)=({np.degrees(roll_after):.2f},{np.degrees(pitch_after):.2f},{np.degrees(yaw_after):.2f})°")
                    print(f"  (roll={np.degrees(CHON_ROLL):.2f}° × {CHON_DURATION}s)")
                    if ENABLE_POSITION_VELOCITY_PD:
                        print(f"  位置・速度PD: 有効 (kp_x={KP_X:.2f} kd_x={KD_X:.2f} kp_y={KP_Y:.2f} kd_y={KD_Y:.2f})")
                    print(f"\n→ スイープ: CHON_DURATION=0.2,0.3,0.4,0.5 で「長さ→Δx」の感度を取る。")
                    break

            if step % 240 == 0:
                r, pitch, y = state.get('roll', 0), state.get('pitch', 0), state.get('yaw', 0)
                line = f"  t={t:.1f}s  {phase}  x={x*100:.1f} y={y*100:.1f} z={z:.2f}  r={np.degrees(r):.1f}° p={np.degrees(pitch):.1f}° y={np.degrees(y):.1f}°"
                if phase == "hover_measure":
                    hi = state.get('height_integral', 0)
                    ho = state.get('height_output', 0)
                    line += f"  hi={hi:.3f} ho={ho:.3f}"
                print(line)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n⏸️ 中断")
    p.disconnect()
    print("✅ シミュレーション終了")


if __name__ == "__main__":
    main()
