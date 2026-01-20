import pybullet as p
import pybullet_data
import numpy as np
import time

# --- 1. GUIモードで接続 ---
print("🚀 PyBulletをGUIモードで起動します...")
device_id = p.connect(p.GUI)
if device_id < 0:
    print("❌ GUIモードでの接続に失敗しました")
    exit(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
print("✅ 床をロードしました")

# --- 2. Vision60ロボットを低い位置からスポーン ---
spawn_height = 0.3  # 低い位置から開始
print(f"📦 Vision60ロボットを高さ {spawn_height}m からスポーンします...")
robot_id = p.loadURDF("quadruped/vision60.urdf", [0, 0, spawn_height])
print("✅ Vision60ロボットをロードしました")

# 仕様書に基づくジョイントマッピング
joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
print(f"📊 可動ジョイント数: {len(joint_indices)} (Abduction, Hip, Knee × 4脚)")

# --- 3. 膝を少し曲げて安定させる ---
knee_angle = 1.0  # 膝の角度（ラジアン）
print(f"🦵 膝を {knee_angle:.2f} rad に曲げて安定させます...")
for i, j_idx in enumerate(joint_indices):
    if i in [2, 5, 8, 11]:  # Kneeジョイント（リスト内のインデックス 2, 5, 8, 11）
        p.resetJointState(robot_id, j_idx, knee_angle)
        print(f"  ジョイント {j_idx} (Knee) を {knee_angle:.2f} rad に設定")
    else:
        p.resetJointState(robot_id, j_idx, 0.0)

# --- 4. 安定を待つ（50～500ステップ） ---
stability_steps = 500  # 安定を待つステップ数
print(f"\n⏳ {stability_steps}ステップ安定を待ちます...")
print("   ロボットが立っていられるか確認します\n")

# 状態を記録するリスト
height_history = []
roll_history = []
pitch_history = []

for step in range(stability_steps):
    # 物理シミュレーションを1ステップ進める
    p.stepSimulation()
    
    # 状態を取得
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    
    height = pos[2]
    roll = euler[0]
    pitch = euler[1]
    
    # 状態を記録
    height_history.append(height)
    roll_history.append(roll)
    pitch_history.append(pitch)
    
    # 一定間隔で状態を表示
    if step % 50 == 0 or step < 10:
        print(f"Step {step:3d}: 高さ={height:.3f}m, Roll={roll:.3f}rad, Pitch={pitch:.3f}rad")
    
    # リアルタイム表示のため少し待つ
    time.sleep(1.0 / 240.0)
    
    # 転倒判定（高さが低すぎる、または傾きが大きすぎる）
    if height < 0.2 or abs(roll) > 0.8 or abs(pitch) > 0.8:
        print(f"\n❌ Step {step} で転倒を検出しました！")
        print(f"   高さ: {height:.3f}m, Roll: {roll:.3f}rad, Pitch: {pitch:.3f}rad")
        break

# --- 5. 結果の確認と表示 ---
print("\n" + "="*60)
print("📊 安定性確認結果")
print("="*60)

if len(height_history) == stability_steps:
    print("✅ 全ステップ完了しました！")
else:
    print(f"⚠️  {len(height_history)}ステップで終了しました")

# 統計情報を表示
final_height = height_history[-1]
final_roll = roll_history[-1]
final_pitch = pitch_history[-1]

avg_height = np.mean(height_history)
avg_roll = np.mean(np.abs(roll_history))
avg_pitch = np.mean(np.abs(pitch_history))

print(f"\n最終状態:")
print(f"  高さ: {final_height:.3f}m")
print(f"  Roll: {final_roll:.3f}rad ({np.degrees(final_roll):.1f}度)")
print(f"  Pitch: {final_pitch:.3f}rad ({np.degrees(final_pitch):.1f}度)")

print(f"\n平均状態:")
print(f"  平均高さ: {avg_height:.3f}m")
print(f"  平均Roll絶対値: {avg_roll:.3f}rad ({np.degrees(avg_roll):.1f}度)")
print(f"  平均Pitch絶対値: {avg_pitch:.3f}rad ({np.degrees(avg_pitch):.1f}度)")

# 安定性の判定
is_stable = (final_height > 0.25 and 
             abs(final_roll) < 0.5 and 
             abs(final_pitch) < 0.5 and
             avg_roll < 0.3 and
             avg_pitch < 0.3)

if is_stable:
    print("\n✅ ロボットは安定して立っていられます！")
else:
    print("\n❌ ロボットは不安定です。初期姿勢やパラメータを調整してください。")

print("\n💡 ヒント:")
print("   - 膝の角度を調整: knee_angle を変更（0.5～1.5程度）")
print("   - スポーン高さを調整: spawn_height を変更（0.2～0.4程度）")
print("   - 他のジョイントも初期化: AbductionやHipの角度も調整可能")

print("\n⏸️  Enterキーを押すと終了します...")
input()

p.disconnect()
print("✅ シミュレーションを終了しました")

