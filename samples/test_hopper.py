"""
hopper.urdfのジョイント動作確認スクリプト

GUIモードで環境を表示し、ジョイントを動かして動作を確認します。
"""
import pybullet as p
import pybullet_data
import numpy as np
import time

# --- 1. GUIモードで接続 ---
print("🚀 PyBulletをGUIモードで起動します...")
client_id = p.connect(p.GUI)
if client_id < 0:
    print("❌ GUIモードでの接続に失敗しました")
    exit(1)

print("✅ PyBullet接続成功")

# --- 2. 環境の設定 ---
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 床をロード
plane_id = p.loadURDF("plane.urdf")
print("✅ 床をロードしました")

# hopper.urdfをロード（少し高い位置から開始）
robot_id = p.loadURDF("hopper.urdf", [0, 0, 1.0])
print("✅ hopper.urdfをロードしました")

# ジョイント情報を取得
num_joints = p.getNumJoints(robot_id)
print(f"📊 ジョイント数: {num_joints}")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    print(f"  ジョイント {i}: {joint_name} (タイプ: {joint_type})")

# --- 3. 環境安定化（50ステップ） ---
print("\n⏳ 環境を安定化しています（50ステップ）...")
for i in range(50):
    p.stepSimulation()
    if i % 10 == 0:
        print(f"  ステップ {i}/50")
print("✅ 安定化完了\n")

# --- 4. 転倒するまで待つ（足は動かさない） ---
print("⏳ 転倒するまで待機中（足は動かしません）...")
fallen = False
step_count = 0

while not fallen:
    time.sleep(1.0 / 240.0)
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    p.stepSimulation()
    step_count += 1
    
    # 転倒を検出（z座標が低くなったら転倒とみなす）
    if pos[2] < 0.5:
        fallen = True
        print(f"✅ 転倒を検出しました（ステップ {step_count}、位置 z={pos[2]:.3f}）")
        print("   これから足をバタバタさせます...\n")
        break
    
    if step_count % 240 == 0:  # 1秒ごと
        print(f"  待機中... ステップ {step_count}, 位置 z={pos[2]:.3f}")

# --- 5. 転倒後に足をバタバタさせる ---
print("🦵 足をバタバタさせます（Ctrl+Cで終了）...")
print("   ジョイント0（knee）を周期的に動かします\n")

step_count = 0
while True:
    # 物理演算の1ステップあたりの時間を考慮して少し待つ
    time.sleep(1.0 / 240.0)  # 240Hzでシミュレーション
    
    # 現在の位置とジョイント状態を取得
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    joint_state = p.getJointState(robot_id, 0)
    
    # 周期的な動き（サイン波を使用）
    # -1.57から1.57ラジアン（約-90度から90度）の範囲で動かす
    t = step_count * 0.01  # 時間パラメータ
    target_angle = np.sin(t) * 1.57  # -1.57から1.57の範囲で振動
    
    # ジョイント0（knee）を位置制御で動かす
    p.setJointMotorControl2(
        robot_id,
        0,  # ジョイントインデックス
        p.POSITION_CONTROL,
        targetPosition=target_angle,
        maxVelocity=5.0  # 最大速度
    )
    
    # 物理シミュレーションを1ステップ進める
    p.stepSimulation()

    step_count += 1
    
    # 定期的に状態を表示
    if step_count % 240 == 0:  # 1秒ごと
        print(f"  ステップ {step_count}: 位置 z={pos[2]:.3f}, ジョイント角度={joint_state[0]:.3f} rad ({np.degrees(joint_state[0]):.1f}°)")
