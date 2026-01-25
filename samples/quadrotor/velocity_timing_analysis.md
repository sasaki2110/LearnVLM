# 速度取得タイミングの分析

速度が「n-1」か「n」かを明確にする。

---

## 制御ループの実行順序

### valcheck.py と forward_quadrotor.py の共通フロー

```python
while True:
    # 1. 目標値設定
    controller.target_pitch = ...
    
    # 2. 制御更新（この中で速度を取得）
    thrusts, state = controller.update(dt)  # ← getBaseVelocity()を呼ぶ
    
    # 3. 推力を適用
    controller.apply_thrusts(thrusts)
    
    # 4. 物理シミュレーションを進める
    p.stepSimulation()  # ← ここで状態が更新される
    
    # 5. ログ出力（位置変化から速度を計算）
    # 前回の位置と今回の位置の差から速度を計算
```

---

## 速度取得のタイミング

### タイミング1: controller.update()内での速度取得

**コード位置**:
- `valcheck.py` 行81: `vel, ang_vel = p.getBaseVelocity(self.robot_id)`
- `forward_quadrotor.py` 行103: `vel, ang_vel = p.getBaseVelocity(self.robot_id)`

**タイミング**: `stepSimulation()`の**前**

**これは何の状態か？**
- **n-1 の stepSimulation()後の状態**
- つまり、**前回のループ**で`stepSimulation()`が実行された後の状態

### タイミング2: ログ出力時での速度取得（一部の場合）

**コード位置**:
- `valcheck.py` 行270: `vel_world_raw = p.getBaseVelocity(robot_id)[0]`
- `forward_quadrotor.py` 行473, 582: `vel_world = p.getBaseVelocity(robot_id)[0]`

**タイミング**: `stepSimulation()`の**後**

**これは何の状態か？**
- **n の stepSimulation()後の状態**
- つまり、**今回のループ**で`stepSimulation()`が実行された後の状態

---

## ログに出力されている速度は？

### valcheck.py のログ

**ログに出力されている速度**:
1. **PyBullet速度（stateから取得）**: `state.get('vel_x_body', 0)`
   - これは`controller.update()`内で取得した速度
   - **タイミング**: `stepSimulation()`の前
   - **状態**: **n-1**

2. **PyBullet速度（デバッグログ）**: `p.getBaseVelocity(robot_id)[0]`
   - これはログ出力時点で取得した速度
   - **タイミング**: `stepSimulation()`の後
   - **状態**: **n**

3. **位置変化から計算した速度**: `(x - prev_x) / dt_log`
   - これは前回の位置と今回の位置の差から計算
   - **タイミング**: `stepSimulation()`の後（位置は更新済み）
   - **状態**: **n と n-1 の間の平均速度**（実際にはnの位置 - n-1の位置）

**結論**: 
- `state`から取得した速度は **n-1**
- デバッグログで取得した速度は **n**
- 位置変化から計算した速度は **n と n-1 の間**

### forward_quadrotor.py のログ

**ログに出力されている速度**:
1. **PyBullet速度（stateから取得）**: `state.get('vel_x_body', 0)`
   - これは`controller.update()`内で取得した速度
   - **タイミング**: `stepSimulation()`の前
   - **状態**: **n-1**

2. **PyBullet速度（デバッグログ）**: `p.getBaseVelocity(robot_id)[0]`
   - これはログ出力時点で取得した速度
   - **タイミング**: `stepSimulation()`の後
   - **状態**: **n**

3. **位置変化から計算した速度**: `(x - prev_x_debug) / dt_debug`
   - これは前回の位置と今回の位置の差から計算
   - **タイミング**: `stepSimulation()`の後（位置は更新済み）
   - **状態**: **n と n-1 の間の平均速度**

**結論**: 
- `state`から取得した速度は **n-1**
- デバッグログで取得した速度は **n**
- 位置変化から計算した速度は **n と n-1 の間**

---

## 問題の本質

### 速度と位置のタイミングのずれ

```
ループ n:
  1. controller.update() → getBaseVelocity() → vel_n-1 を取得
  2. apply_thrusts() → 力を適用
  3. stepSimulation() → 位置・速度を更新 → pos_n, vel_n が計算される
  4. ログ出力:
     - stateから取得した速度: vel_n-1 (n-1の状態)
     - 位置変化から計算: (pos_n - pos_n-1) / dt (nとn-1の間)
     - デバッグログで取得: vel_n (nの状態)
```

**問題点**:
- `state`から取得した速度（vel_n-1）と位置変化から計算した速度（(pos_n - pos_n-1) / dt）を比較している
- しかし、vel_n-1は「n-1の状態」で、(pos_n - pos_n-1) / dtは「nとn-1の間の平均速度」
- これらは**異なるタイミング**の速度なので、一致しないのは当然

### 正しい比較方法

速度と位置変化を比較するには、**同じタイミング**の速度を使う必要がある：

1. **n-1の速度とn-1からnへの位置変化を比較**
   - vel_n-1 vs (pos_n - pos_n-1) / dt
   - ただし、これは「n-1の瞬間速度」vs「n-1からnへの平均速度」なので、加速度がある場合は一致しない

2. **nの速度とnからn+1への位置変化を比較**
   - vel_n vs (pos_n+1 - pos_n) / dt
   - これも「nの瞬間速度」vs「nからn+1への平均速度」なので、加速度がある場合は一致しない

3. **台形公式を使った比較**
   - (vel_n-1 + vel_n) / 2 vs (pos_n - pos_n-1) / dt
   - これは「n-1からnへの平均速度」vs「n-1からnへの平均速度」なので、理論的には一致するはず

---

## 結論

### valcheck.py のログの速度

- **stateから取得した速度**: **n-1** ✅
- **デバッグログで取得した速度**: **n** ✅
- **位置変化から計算した速度**: **n と n-1 の間** ✅

### forward_quadrotor.py のログの速度

- **stateから取得した速度**: **n-1** ✅
- **デバッグログで取得した速度**: **n** ✅
- **位置変化から計算した速度**: **n と n-1 の間** ✅

### この考え方は正しいか？

**はい、この考え方は正しいです。**

ただし、速度と位置変化を比較する際は、**同じタイミング**の速度を使う必要があります。

現在のログでは：
- `state`から取得した速度（n-1）と位置変化から計算した速度（nとn-1の間）を比較している
- これらは異なるタイミングなので、加速度がある場合は一致しないのは当然

**解決策**:
- 台形公式を使う: (vel_n-1 + vel_n) / 2 と (pos_n - pos_n-1) / dt を比較
- または、nの速度とnからn+1への位置変化を比較（次のループで）

---

## 参考

- `control_flow_sequence.md`: 制御フローのシーケンス図
- `valcheck_log.md`: 検証結果の記録
