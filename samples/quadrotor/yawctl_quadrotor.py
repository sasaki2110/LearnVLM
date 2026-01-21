"""
Quadrotor PID制御による高度な飛行制御（Yaw制御 + 位置制御）

飛行パターン:
1. 初期段階 3m地点で10秒ホバリング
2. 前へ2m前進
3. 初期位置を中心点として、半径2mの円を描くように旋回
4. 初期位置へ戻ってホバリング
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
# Quadrotor制御クラス（拡張版）
# ============================================================================

class QuadrotorController:
    """QuadrotorのPID制御（位置制御 + Yaw制御付き）"""
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # プロペラの位置（URDFから取得した値）
        # prop1: 右, prop2: 前, prop3: 左, prop4: 後
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
        
        # 位置制御（X、Y方向）- 速度目標を出力
        self.x_pid = PIDController(kp=0.8, ki=0.01, kd=0.3)
        self.y_pid = PIDController(kp=0.8, ki=0.01, kd=0.3)
        
        # 速度制御（X、Y方向）- 姿勢目標を出力
        self.vel_x_pid = PIDController(kp=1.5, ki=0.05, kd=0.5)
        self.vel_y_pid = PIDController(kp=1.5, ki=0.05, kd=0.5)
        
        # 姿勢制御（Roll, Pitch, Yaw）
        self.roll_pid = PIDController(kp=10.0, ki=0.5, kd=5.0)
        self.pitch_pid = PIDController(kp=10.0, ki=0.5, kd=5.0)
        self.yaw_pid = PIDController(kp=8.0, ki=0.2, kd=3.0)  # Yaw制御を強化
        
        # 目標値
        self.target_height = 3.0
        self.target_x = 0.0
        self.target_y = 0.0
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
        
        # 速度から姿勢への変換ゲイン
        self.velocity_to_attitude_gain = 0.15  # 速度目標からRoll/Pitchへの変換
        
        # Roll/Pitchの最大角度制限（ラジアン）
        self.max_roll_pitch = 0.3  # 約17度
        
        # 位置制御の有効/無効フラグ（デバッグ用）
        self.position_control_enabled = True
    
    def update(self, dt):
        """
        制御を更新してプロペラの推力を計算
        
        Returns:
            tuple: (thrusts, state_dict)
        """
        # 現在の状態を取得
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # オイラー角に変換
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        
        # 速度
        vel_z = vel[2]
        vel_x = vel[0]
        vel_y = vel[1]
        roll_vel = ang_vel[0]
        pitch_vel = ang_vel[1]
        yaw_vel = ang_vel[2]
        
        # カスケード制御: 位置 → 速度 → 姿勢
        if self.position_control_enabled:
            # 1. 位置制御: 位置誤差から速度目標を計算
            # PID制御: target_vel = PID(target_position, current_position, current_velocity, dt)
            target_vel_x = self.x_pid.compute(self.target_x, pos[0], vel_x, dt)
            target_vel_y = self.y_pid.compute(self.target_y, pos[1], vel_y, dt)
            
            # 速度目標を制限（安全のため）
            max_vel = 0.8  # m/s
            target_vel_x = np.clip(target_vel_x, -max_vel, max_vel)
            target_vel_y = np.clip(target_vel_y, -max_vel, max_vel)
            
            # 2. 速度制御: 速度誤差から姿勢目標を計算
            # PID制御: target_attitude = PID(target_velocity, current_velocity, 0, dt) * gain
            # PyBulletの座標系では、X軸が前後方向、Y軸が左右方向の可能性が高い
            # X軸正方向（前）に進むには前傾（正のPitch）が必要
            # Pitchが正 → 前傾（前が下がる）→ X軸正方向（前）に進む
            target_pitch_from_vel = self.vel_x_pid.compute(target_vel_x, vel_x, 0, dt) * self.velocity_to_attitude_gain
            # Y軸正方向（右）に進むには右傾（正のRoll）が必要
            # Rollが正 → 右に傾く → Y軸正方向（右）に進む
            target_roll_from_vel = self.vel_y_pid.compute(target_vel_y, vel_y, 0, dt) * self.velocity_to_attitude_gain
            
            # 3. 姿勢制御の目標値（速度制御からの入力）
            target_roll = self.target_roll + target_roll_from_vel
            target_pitch = self.target_pitch + target_pitch_from_vel
        else:
            # 位置制御を無効化（姿勢制御のみ）
            target_roll = self.target_roll
            target_pitch = self.target_pitch
        
        # Roll/Pitchの最大角度を制限
        target_roll = np.clip(target_roll, -self.max_roll_pitch, self.max_roll_pitch)
        target_pitch = np.clip(target_pitch, -self.max_roll_pitch, self.max_roll_pitch)
        
        # PID制御で各軸の制御出力を計算
        height_output = self.height_pid.compute(self.target_height, pos[2], vel_z, dt)
        roll_output = self.roll_pid.compute(target_roll, roll, roll_vel, dt)
        pitch_output = self.pitch_pid.compute(target_pitch, pitch, pitch_vel, dt)
        yaw_output = self.yaw_pid.compute(self.target_yaw, yaw, yaw_vel, dt)
        
        # 各プロペラの推力を計算
        # 高さ制御: 全プロペラに同じ推力
        thrust_height = self.base_thrust + height_output / 4.0
        
        # Roll制御: prop1とprop3の差（右と左）
        # 制限を追加して、高さ制御を優先
        thrust_roll = roll_output / (2.0 * self.arm_length)
        thrust_roll = np.clip(thrust_roll, -self.base_thrust * 0.5, self.base_thrust * 0.5)  # 基本推力の50%以内
        
        # Pitch制御: prop2とprop4の差（前と後）
        # 制限を追加して、高さ制御を優先
        thrust_pitch = pitch_output / (2.0 * self.arm_length)
        thrust_pitch = np.clip(thrust_pitch, -self.base_thrust * 0.5, self.base_thrust * 0.5)  # 基本推力の50%以内
        
        # Yaw制御: モーメント係数を使って推力差を生成
        # 正のYaw出力（時計回り）→ 正のモーメントを持つプロペラを減らし、負のモーメントを持つプロペラを増やす
        yaw_scale = 0.15  # Yaw制御のスケール（調整可能）
        thrust_yaw = yaw_output * yaw_scale
        thrust_yaw = np.clip(thrust_yaw, -self.base_thrust * 0.2, self.base_thrust * 0.2)  # 基本推力の20%以内
        
        # 各プロペラの推力
        # prop1: 右, prop2: 前, prop3: 左, prop4: 後
        thrusts = [
            thrust_height + thrust_roll + thrust_yaw * self.prop_moments[0],  # prop1 (右)
            thrust_height + thrust_pitch + thrust_yaw * self.prop_moments[1],  # prop2 (前)
            thrust_height - thrust_roll + thrust_yaw * self.prop_moments[2],   # prop3 (左)
            thrust_height - thrust_pitch + thrust_yaw * self.prop_moments[3]   # prop4 (後)
        ]
        
        # 推力の制限（高さ制御を優先）
        # 各プロペラの推力が基本推力の30%以下にならないようにする（姿勢制御の柔軟性を確保）
        min_thrust_per_prop = self.base_thrust * 0.3
        thrusts = [np.clip(t, min_thrust_per_prop, self.max_thrust) for t in thrusts]
        
        return thrusts, {
            'position': pos,
            'velocity': vel,
            'height': pos[2],
            'x': pos[0],
            'y': pos[1],
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'height_error': self.target_height - pos[2],
            'x_error': self.target_x - pos[0],
            'y_error': self.target_y - pos[1],
            'roll_error': target_roll - roll,
            'pitch_error': target_pitch - pitch,
            'yaw_error': self.target_yaw - yaw
        }
    
    def apply_thrusts(self, thrusts):
        """プロペラの推力を適用"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 各プロペラの位置に個別に力を適用
        for i, (thrust, prop_pos_local) in enumerate(zip(thrusts, self.prop_positions)):
            # ローカル座標系での力（上向き）
            force_local = [0, 0, thrust]
            
            # ワールド座標系に変換
            force_world = p.rotateVector(orn, force_local)
            if isinstance(force_world, tuple):
                force_world = list(force_world)
            
            # プロペラの位置をワールド座標系に変換
            prop_pos_local_list = list(prop_pos_local) if isinstance(prop_pos_local, np.ndarray) else prop_pos_local
            
            prop_pos_world, _ = p.multiplyTransforms(
                pos, orn,
                prop_pos_local_list, [0, 0, 0, 1]
            )
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
    
    def reset(self):
        """PID制御器をリセット"""
        self.height_pid.reset()
        self.x_pid.reset()
        self.y_pid.reset()
        self.vel_x_pid.reset()
        self.vel_y_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()


# ============================================================================
# 飛行フェーズ管理
# ============================================================================

class FlightMission:
    """飛行ミッション管理"""
    
    def __init__(self, controller):
        self.controller = controller
        self.phase = 0
        self.phase_start_time = 0.0
        self.initial_position = [0.0, 0.0, 0.0]
        self.circle_center = [0.0, 0.0]
        self.circle_radius = 2.0
        self.circle_angle = 0.0
        self.circle_speed = 0.08  # rad/s（円軌道の角速度、非常に遅くして追従しやすく）
        self.circle_start_position = [0.0, 0.0]  # 円軌道開始時の位置
        self.enable_circle = False  # 円軌道を有効にするか（Falseでスキップ、Trueで有効化）
        self.phase2_completed = False  # Phase 2完了フラグ
        
        # 軌道生成用
        self.current_target_x = 0.0
        self.current_target_y = 0.0
        self.target_x_final = 0.0
        self.target_y_final = 0.0
        self.trajectory_speed = 0.5  # m/s（軌道の速度、円軌道に追従できるように調整）
        self.phase_started = False  # フェーズ開始フラグ
        
    def update(self, current_time, state):
        """飛行フェーズを更新"""
        dt = current_time - self.phase_start_time
        
        if self.phase == 0:
            # Phase 1: 3m地点で10秒ホバリング
            self.controller.target_height = 3.0
            self.controller.target_x = self.initial_position[0]
            self.controller.target_y = self.initial_position[1]
            self.controller.target_yaw = 0.0
            
            if dt >= 10.0:
                print("✅ Phase 1 完了: 10秒ホバリング")
                self.phase = 1
                self.phase_start_time = current_time
                self.phase_started = False  # フェーズ開始フラグをリセット
                self.controller.reset()  # PIDをリセット
        
        elif self.phase == 1:
            # Phase 2: 前へ2m前進（軌道生成を使用）
            self.controller.target_height = 3.0
            self.controller.target_yaw = 0.0
            
            # 最終目標位置
            self.target_x_final = self.initial_position[0] + 2.0
            self.target_y_final = self.initial_position[1]
            
            # 軌道生成: 現在位置から目標位置へ段階的に移動
            if not self.phase_started:  # フェーズ開始時
                self.current_target_x = state['x']
                self.current_target_y = state['y']
                self.phase_started = True
            
            # 目標位置までの距離
            dx = self.target_x_final - self.current_target_x
            dy = self.target_y_final - self.current_target_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # 軌道に沿って目標位置を更新（段階的に近づける）
            if distance > 0.1:  # まだ目標に到達していない
                # 方向ベクトル
                direction_x = dx / distance if distance > 0 else 0
                direction_y = dy / distance if distance > 0 else 0
                
                # 1ステップで移動する距離
                step_distance = self.trajectory_speed * (1.0 / 240.0)  # dt
                
                # 目標位置を更新
                if step_distance < distance:
                    self.current_target_x += direction_x * step_distance
                    self.current_target_y += direction_y * step_distance
                else:
                    # 最後のステップ
                    self.current_target_x = self.target_x_final
                    self.current_target_y = self.target_y_final
            else:
                # 目標位置に到達
                self.current_target_x = self.target_x_final
                self.current_target_y = self.target_y_final
            
            # 制御器に目標位置を設定
            self.controller.target_x = self.current_target_x
            self.controller.target_y = self.current_target_y
            
            # 最終目標位置からの距離を計算
            final_x_error = self.target_x_final - state['x']
            final_y_error = self.target_y_final - state['y']
            final_distance = np.sqrt(final_x_error**2 + final_y_error**2)
            
            # 目標位置に到達したか確認
            # 条件: X方向に2m以上進んだ、または最終目標位置に近づいた（0.3m以内）、かつ最低2秒経過
            vel_x = state['velocity'][0]
            vel_y = state['velocity'][1]
            speed_xy = np.sqrt(vel_x**2 + vel_y**2)
            
            # X方向の進捗を確認（初期位置から2m以上進んだか）
            x_progress = state['x'] - self.initial_position[0]
            
            # 最低2秒は前進を試みる（到達判定が早すぎるのを防ぐ）
            if dt >= 2.0:
                # X方向に2m以上進んだ、または最終目標位置に近づいた
                if x_progress >= 1.8 or (final_distance < 0.3 and speed_xy < 0.5):
                    print(f"✅ Phase 2 完了: 2m前進 (到達位置: {state['x']:.2f}, {state['y']:.2f}, X進捗: {x_progress:.2f}m, 経過時間: {dt:.1f}s)")
                    # 円軌道の中心を現在位置に設定（初期位置を中心に半径2mの円）
                    # 円の中心は初期位置(0, 0)のまま
                    self.circle_center = [self.initial_position[0], self.initial_position[1]]
                    # 円軌道開始時の位置を記録
                    self.circle_start_position = [state['x'], state['y']]
                    # 現在位置から円軌道への開始角度を計算
                    dx = self.circle_start_position[0] - self.circle_center[0]
                    dy = self.circle_start_position[1] - self.circle_center[1]
                    self.circle_angle = np.arctan2(dy, dx)  # 現在位置の角度から開始
                    print(f"   円軌道開始: 中心({self.circle_center[0]:.2f}, {self.circle_center[1]:.2f}), 開始角度: {np.degrees(self.circle_angle):.1f}°")
                    self.phase = 2
                    self.phase_start_time = current_time
                    self.phase_started = False  # フェーズ開始フラグをリセット
                    # PIDをリセットして円軌道に移行
                    self.controller.reset()
        
        elif self.phase == 2:
            # Phase 3: 半径2mの円を描くように旋回（またはスキップ）
            if not self.enable_circle:
                # 円軌道をスキップしてPhase 4へ
                print("⏭️  円軌道をスキップします")
                self.phase = 3
                self.phase_start_time = current_time
                self.phase_started = False
                self.controller.reset()
            else:
                self.controller.target_height = 3.0
                
                # 円軌道の計算
                dt_actual = current_time - self.phase_start_time
                # 開始角度から時計回りに回転
                current_circle_angle = self.circle_angle + self.circle_speed * dt_actual
                
                # 円の周りを1周（2π）
                angle_progress = current_circle_angle - self.circle_angle
                if angle_progress >= 2 * np.pi:
                    print("✅ Phase 3 完了: 円軌道1周")
                    self.phase = 3
                    self.phase_start_time = current_time
                    self.phase_started = False  # フェーズ開始フラグをリセット
                    self.controller.reset()
                else:
                    # 円軌道上の目標位置を計算
                    circle_center_x = self.circle_center[0]
                    circle_center_y = self.circle_center[1]
                    target_x_on_circle = circle_center_x + self.circle_radius * np.cos(current_circle_angle)
                    target_y_on_circle = circle_center_y + self.circle_radius * np.sin(current_circle_angle)
                    
                    # 軌道生成: 現在位置から円軌道上の目標位置へ段階的に移動
                    if not self.phase_started:  # フェーズ開始時
                        self.current_target_x = state['x']
                        self.current_target_y = state['y']
                        self.phase_started = True
                    
                    # 円軌道上の目標位置までの距離
                    dx = target_x_on_circle - self.current_target_x
                    dy = target_y_on_circle - self.current_target_y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # 軌道に沿って目標位置を更新（段階的に近づける）
                    # 円軌道の場合は、直接円軌道上の目標位置を使用（追従を優先）
                    # ただし、距離が大きすぎる場合は段階的に近づける
                    if distance > 0.5:  # 目標位置が遠い場合は段階的に近づける
                        # 方向ベクトル
                        direction_x = dx / distance if distance > 0 else 0
                        direction_y = dy / distance if distance > 0 else 0
                        
                        # 1ステップで移動する距離（円軌道に追従できる速度）
                        step_distance = self.trajectory_speed * (1.0 / 240.0)  # dt
                        
                        # 目標位置を更新
                        if step_distance < distance:
                            self.current_target_x += direction_x * step_distance
                            self.current_target_y += direction_y * step_distance
                        else:
                            # 最後のステップ
                            self.current_target_x = target_x_on_circle
                            self.current_target_y = target_y_on_circle
                    else:
                        # 目標位置が近い場合は、直接円軌道上の目標位置を使用
                        self.current_target_x = target_x_on_circle
                        self.current_target_y = target_y_on_circle
                    
                    # 制御器に目標位置を設定
                    self.controller.target_x = self.current_target_x
                    self.controller.target_y = self.current_target_y
                    
                    # Yawを円の接線方向に向ける（進行方向）
                    self.controller.target_yaw = current_circle_angle + np.pi / 2.0
                
                # 制御器に目標位置を設定
                self.controller.target_x = self.current_target_x
                self.controller.target_y = self.current_target_y
                
                # Yawを円の接線方向に向ける（進行方向）
                self.controller.target_yaw = current_circle_angle + np.pi / 2.0
        
        elif self.phase == 3:
            # Phase 4: 初期位置へ戻ってホバリング（軌道生成を使用）
            self.controller.target_height = 3.0
            self.controller.target_yaw = 0.0
            
            # 最終目標位置
            self.target_x_final = self.initial_position[0]
            self.target_y_final = self.initial_position[1]
            
            # 軌道生成: 現在位置から目標位置へ段階的に移動
            if not self.phase_started:  # フェーズ開始時
                self.current_target_x = state['x']
                self.current_target_y = state['y']
                self.phase_started = True
            
            # 目標位置までの距離
            dx = self.target_x_final - self.current_target_x
            dy = self.target_y_final - self.current_target_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # 軌道に沿って目標位置を更新（段階的に近づける）
            if distance > 0.1:  # まだ目標に到達していない
                # 方向ベクトル
                direction_x = dx / distance if distance > 0 else 0
                direction_y = dy / distance if distance > 0 else 0
                
                # 1ステップで移動する距離
                step_distance = self.trajectory_speed * (1.0 / 240.0)  # dt
                
                # 目標位置を更新
                if step_distance < distance:
                    self.current_target_x += direction_x * step_distance
                    self.current_target_y += direction_y * step_distance
                else:
                    # 最後のステップ
                    self.current_target_x = self.target_x_final
                    self.current_target_y = self.target_y_final
            else:
                # 目標位置に到達
                self.current_target_x = self.target_x_final
                self.current_target_y = self.target_y_final
            
            # 制御器に目標位置を設定
            self.controller.target_x = self.current_target_x
            self.controller.target_y = self.current_target_y
            
            # 目標位置に到達したか確認（誤差0.2m以内、かつ速度が小さい）
            vel_x = state['velocity'][0]
            vel_y = state['velocity'][1]
            speed_xy = np.sqrt(vel_x**2 + vel_y**2)
            
            if abs(state['x_error']) < 0.2 and abs(state['y_error']) < 0.2 and speed_xy < 0.3:
                if dt < 0.1:  # 初回のみメッセージを表示
                    print("✅ Phase 4 完了: 初期位置に戻りました")
                    print("🎉 全ミッション完了！ホバリングを継続します...")
                # Phase 4のまま継続（ホバリング）
        
        return self.phase


# ============================================================================
# メイン関数
# ============================================================================

def main():
    """メイン関数"""
    print("=" * 60)
    print("🚁 Quadrotor 高度な飛行制御（Yaw制御 + 位置制御）")
    print("=" * 60)
    print("\n📋 飛行ミッション:")
    print("   1. 3m地点で10秒ホバリング")
    print("   2. 前へ2m前進")
    print("   3. 初期位置を中心に半径2mの円を描くように旋回")
    print("   4. 初期位置へ戻ってホバリング")
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
    
    # 飛行ミッションを初期化
    mission = FlightMission(controller)
    mission.initial_position = [0.0, 0.0, 0.0]
    mission.circle_center = [0.0, 0.0]
    
    print(f"\n📊 初期設定:")
    print(f"   初期位置: ({mission.initial_position[0]}, {mission.initial_position[1]}, {mission.initial_position[2]})")
    print(f"   円の中心: ({mission.circle_center[0]}, {mission.circle_center[1]})")
    print(f"   円の半径: {mission.circle_radius}m")
    
    print(f"\n🎮 制御を開始します...")
    print(f"   GUIでドローンの動作を確認してください")
    print(f"   Ctrl+Cで終了\n")
    
    # シミュレーションループ
    dt = 1.0 / 240.0  # PyBulletの標準タイムステップ
    step_count = 0
    current_time = 0.0
    
    try:
        while True:
            # PID制御を更新
            thrusts, state = controller.update(dt)
            
            # 飛行ミッションを更新
            phase = mission.update(current_time, state)
            
            # 推力を適用
            controller.apply_thrusts(thrusts)
            
            # シミュレーションを1ステップ進める
            p.stepSimulation()
            
            # 状態を表示（1秒ごと）
            if step_count % 240 == 0:
                phase_names = ["ホバリング", "前進", "円軌道", "帰還"]
                vel_x = state['velocity'][0]
                vel_y = state['velocity'][1]
                speed_xy = np.sqrt(vel_x**2 + vel_y**2)
                total_thrust = sum(thrusts)
                print(f"Time: {current_time:.1f}s | Phase: {phase_names[phase]} | "
                      f"位置: ({state['x']:.2f}, {state['y']:.2f}, {state['height']:.2f})m | "
                      f"速度: {speed_xy:.2f}m/s | "
                      f"Roll: {np.degrees(state['roll']):.1f}° Pitch: {np.degrees(state['pitch']):.1f}° Yaw: {np.degrees(state['yaw']):.1f}° | "
                      f"総推力: {total_thrust:.2f}N | "
                      f"目標: ({controller.target_x:.2f}, {controller.target_y:.2f}, {controller.target_height:.2f})m")
            
            step_count += 1
            current_time += dt
            time.sleep(dt)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  制御を終了します...")
    
    p.disconnect()
    print("✅ シミュレーションを終了しました")


if __name__ == "__main__":
    main()
