# Quadrotor制御フローのシーケンス図

スクリプトからドローンへの指示、PID制御、PyBulletシミュレーションの流れを可視化する。

---

## 基本制御ループのシーケンス図

```mermaid
sequenceDiagram
    participant Script as スクリプト<br/>(main loop)
    participant Controller as QuadrotorController
    participant PID as PID制御器<br/>(height/roll/pitch/yaw)
    participant PyBullet as PyBullet<br/>(物理シミュレーション)
    participant Drone as ドローン<br/>(機体状態)

    Note over Script: 制御ループ開始 (dt = 1/240秒ごと)
    
    Script->>Script: 目標値設定<br/>(target_pitch, target_roll等)
    
    Script->>Controller: controller.update(dt)
    
    Note over Controller: 1. 現在の状態を取得
    Controller->>PyBullet: getBasePositionAndOrientation()
    PyBullet-->>Controller: pos, orn (位置, 姿勢)
    Controller->>PyBullet: getBaseVelocity()
    PyBullet-->>Controller: vel, ang_vel (速度, 角速度)
    
    Note over Controller: 2. PID制御で出力を計算
    Controller->>PID: height_pid.compute(target, current, vel_z, dt)
    PID-->>Controller: height_output
    Controller->>PID: roll_pid.compute(target, current, roll_vel, dt)
    PID-->>Controller: roll_output
    Controller->>PID: pitch_pid.compute(target, current, pitch_vel, dt)
    PID-->>Controller: pitch_output
    Controller->>PID: yaw_pid.compute(target, current, yaw_vel, dt)
    PID-->>Controller: yaw_output
    
    Note over Controller: 3. 推力配分を計算
    Controller->>Controller: 各プロペラの推力計算<br/>(thrusts[4])
    
    Controller-->>Script: thrusts, state
    
    Note over Script: 4. 推力を適用
    Script->>Controller: controller.apply_thrusts(thrusts)
    
    Note over Controller: 各プロペラに力を適用
    loop 各プロペラ (4つ)
        Controller->>PyBullet: applyExternalForce(robot_id, force, position)
    end
    
    Note over Script: 5. 物理シミュレーションを進める
    Script->>PyBullet: stepSimulation()
    
    Note over PyBullet: 物理エンジンが状態を更新
    PyBullet->>Drone: 力とトルクを適用
    Drone->>Drone: 位置・速度・姿勢を更新<br/>(物理法則に従って)
    Drone-->>PyBullet: 新しい状態
    
    Note over Script: 6. 次のループへ
    Script->>Script: step += 1<br/>t = step * dt
```

---

## 速度取得タイミングの問題点

```mermaid
sequenceDiagram
    participant Script as スクリプト
    participant Controller as Controller.update()
    participant PyBullet as PyBullet
    participant Physics as 物理エンジン

    Note over Script: ループ開始 (t = n * dt)
    
    Script->>Controller: update(dt)
    
    Note over Controller: タイミング1: 速度を取得
    Controller->>PyBullet: getBaseVelocity()
    Note right of PyBullet: この時点の速度を返す<br/>(前回のstepSimulation()後の状態)
    PyBullet-->>Controller: vel (速度)
    
    Controller->>Controller: PID制御で推力計算
    
    Controller-->>Script: thrusts
    
    Script->>Controller: apply_thrusts(thrusts)
    Controller->>PyBullet: applyExternalForce() × 4
    
    Script->>PyBullet: stepSimulation()
    
    Note over Physics: タイミング2: 物理エンジンが状態を更新
    Physics->>Physics: 力とトルクを適用<br/>位置・速度を更新
    
    Note over Script: 問題点:<br/>getBaseVelocity()で取得した速度は、<br/>まだ新しい力が適用される前の状態
    Note over Script: stepSimulation()後、速度は更新されるが、<br/>次のループで取得されるまで待つ必要がある
    
    Script->>Script: 次のループ (t = (n+1) * dt)
```

---

## 詳細な制御フロー（forward_quadrotor.pyの場合）

```mermaid
sequenceDiagram
    participant Script as forward_quadrotor.py
    participant Controller as QuadrotorController
    participant PID as PID制御器
    participant PosVelPD as 位置・速度PD<br/>(外側ループ)
    participant PyBullet as PyBullet

    Note over Script: チョン動作中
    
    Script->>Script: 目標値設定<br/>(chon_roll_override = 0.03 rad)
    
    Script->>Controller: controller.update(dt)
    
    Note over Controller: 1. 現在の状態を取得
    Controller->>PyBullet: getBasePositionAndOrientation()
    PyBullet-->>Controller: pos, orn
    Controller->>PyBullet: getBaseVelocity()
    PyBullet-->>Controller: vel, ang_vel
    
    Note over Controller: 2. 座標変換
    Controller->>Controller: ワールド座標系→機体座標系<br/>(vel_x_body, vel_y_body)
    
    Note over Controller: 3. 位置・速度PD（外側ループ）
    alt 位置・速度PDが有効 かつ チョン中でない
        Controller->>PosVelPD: 位置誤差を機体座標系に変換
        Controller->>PosVelPD: 速度D: -kd_x * vel_x_body
        Controller->>PosVelPD: 速度D: -kd_y * vel_y_body
        PosVelPD-->>Controller: roll_offset, pitch_offset
    else チョン中
        Note over Controller: 位置・速度PDは無効化
    end
    
    Note over Controller: 4. 姿勢PID（内側ループ）
    Controller->>PID: roll_pid.compute(target + roll_offset, roll, roll_vel, dt)
    PID-->>Controller: roll_output
    Controller->>PID: pitch_pid.compute(target + pitch_offset, pitch, pitch_vel, dt)
    PID-->>Controller: pitch_output
    
    Note over Controller: 5. 推力配分
    Controller->>Controller: 推力配分計算<br/>(解決策1: 位置・速度PDの出力を直接推力に反映)
    
    Controller-->>Script: thrusts, state
    
    Script->>Controller: apply_thrusts(thrusts)
    Controller->>PyBullet: applyExternalForce() × 4
    
    Script->>PyBullet: stepSimulation()
    
    Note over PyBullet: 物理エンジンが状態を更新<br/>新しい位置・速度が計算される
    
    Script->>Script: 次のループ
```

---

## 速度取得タイミングの問題の詳細

```mermaid
sequenceDiagram
    participant Script as スクリプト
    participant Controller as Controller
    participant PyBullet as PyBullet
    participant Physics as 物理エンジン

    Note over Script: ループ n (t = n * dt)
    
    rect rgb(200, 200, 255)
        Note over Script,Physics: タイミング1: 速度取得
        Script->>Controller: update(dt)
        Controller->>PyBullet: getBaseVelocity()
        Note right of PyBullet: この時点の速度<br/>(ループn-1のstepSimulation()後の状態)
        PyBullet-->>Controller: vel_n-1 (前回の速度)
    end
    
    Controller->>Controller: PID制御で推力計算<br/>(vel_n-1を使用)
    Controller-->>Script: thrusts_n
    
    Script->>Controller: apply_thrusts(thrusts_n)
    Controller->>PyBullet: applyExternalForce()
    
    rect rgb(255, 200, 200)
        Note over Script,Physics: タイミング2: 物理シミュレーション
        Script->>PyBullet: stepSimulation()
        Physics->>Physics: 力とトルクを適用<br/>位置・速度を更新
        Note right of Physics: 新しい速度 vel_n が計算される<br/>しかし、この時点では取得されない
    end
    
    Note over Script: ループ n+1 (t = (n+1) * dt)
    
    rect rgb(200, 255, 200)
        Note over Script,Physics: タイミング3: 次の速度取得
        Script->>Controller: update(dt)
        Controller->>PyBullet: getBaseVelocity()
        Note right of PyBullet: この時点の速度<br/>(ループnのstepSimulation()後の状態)
        PyBullet-->>Controller: vel_n (今回の速度)
    end
    
    Note over Script: 問題点:<br/>vel_n-1を使って推力計算したが、<br/>実際の位置変化はvel_nに基づいている<br/>→ 速度と位置変化の不一致が発生
```

---

## 問題の原因

1. **速度取得のタイミング**
   - `getBaseVelocity()`は`stepSimulation()`の**前**に呼ばれる
   - 取得した速度は**前回の**`stepSimulation()`後の状態
   - しかし、位置変化は**今回の**`stepSimulation()`後の状態

2. **制御ループとのずれ**
   - PID制御は前回の速度に基づいて推力を計算
   - 物理エンジンは新しい力に基づいて状態を更新
   - 次のループで新しい速度を取得するまで、ずれが発生

3. **加速度が大きい時の問題**
   - 加速度が大きいと、速度の変化も大きい
   - タイミングのずれが速度の不一致として現れる
   - 特に方向（符号）が逆になることがある

---

## 参考

- `valcheck.py`: ホバリング状態での速度検証
- `forward_quadrotor.py`: チョン動作中の速度検証
- `valcheck_log.md`: 検証結果の記録
