"""
quad.urdfのジョイント動作確認スクリプト

GUIモードで環境を表示し、ジョイントを動かして動作を確認します。
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# --- 1. 四足ロボットのURDFを生成（必要に応じて） ---
def create_quadruped_urdf():
    """四足ロボットのURDFファイルを生成"""
    urdf_content = """
    <robot name="l_leg_quad">
      <link name="base">
        <visual><geometry><box size="0.4 0.4 0.1"/></geometry><material name="blue"><color rgba="0 0 1 1"/></material></visual>
        <collision><geometry><box size="0.4 0.4 0.1"/></geometry></collision>
        <inertial><mass value="2.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
      </link>
    """
    # 4本の足を生成
    positions = [ [0.2, 0.2], [0.2, -0.2], [-0.2, 0.2], [-0.2, -0.2] ]
    for i, pos in enumerate(positions):
        urdf_content += f"""
      <link name="thigh_{i}">
        <visual><origin xyz="0 0 -0.1"/><geometry><box size="0.05 0.05 0.2"/></geometry><material name="red"><color rgba="1 0 0 1"/></material></visual>
        <collision><origin xyz="0 0 -0.1"/><geometry><box size="0.05 0.05 0.2"/></geometry></collision>
        <inertial>
            <origin xyz="0 0 -0.1"/>
            <mass value="0.2"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
      </link>
      <link name="calf_{i}">
        <visual><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.05 0.05"/></geometry><material name="green"><color rgba="0 1 0 1"/></material></visual>
        <collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.05 0.05"/></geometry></collision>
        <inertial>
            <origin xyz="0.1 0 0"/>
            <mass value="0.2"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
      </link>
      <joint name="hip_{i}" type="revolute">
        <parent link="base"/><child link="thigh_{i}"/><origin xyz="{pos[0]} {pos[1]} 0"/><axis xyz="0 1 0"/>
        <limit effort="100" lower="-1.5" upper="1.5" velocity="10"/>
      </joint>
      <joint name="knee_{i}" type="revolute">
        <parent link="thigh_{i}"/><child link="calf_{i}"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/>
        <limit effort="100" lower="-1.5" upper="1.5" velocity="10"/>
      </joint>
        """
    urdf_content += "</robot>"
    with open("quad.urdf", "w") as f:
        f.write(urdf_content)
    print("✅ quad.urdf を生成しました")

# --- 2. GUIモードで接続 ---
print("🚀 PyBulletをGUIモードで起動します...")
client_id = p.connect(p.GUI)
if client_id < 0:
    print("❌ GUIモードでの接続に失敗しました")
    exit(1)

print("✅ PyBullet接続成功")

# --- 3. 環境の設定 ---
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 床をロード
plane_id = p.loadURDF("plane.urdf")
print("✅ 床をロードしました")

# URDFファイルが存在しない場合は生成
if not os.path.exists("quad.urdf"):
    create_quadruped_urdf()
else:
    print("✅ quad.urdf が見つかりました")

# quad.urdfをロード（学習時と同じ高さから開始）
robot_id = p.loadURDF("quad.urdf", [0, 0, 0.3])
print("✅ quad.urdfをロードしました")

# ジョイント情報を取得
num_joints = p.getNumJoints(robot_id)
print(f"📊 ジョイント数: {num_joints}")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    print(f"  ジョイント {i}: {joint_name} (タイプ: {joint_type})")

# --- 4. 環境安定化（50ステップ） ---
print("\n⏳ 環境を安定化しています（50ステップ）...")
for i in range(50):
    p.stepSimulation()
    if i % 10 == 0:
        print(f"  ステップ {i}/50")
print("✅ 安定化完了\n")

# --- 5. 少し待ってからジョイントを動かす ---
print("⏳ 少し待機してからジョイントを動かします（3秒間）...")
wait_steps = 3 * 240  # 3秒間待機
step_count = 0

while step_count < wait_steps:
    time.sleep(1.0 / 240.0)
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    p.stepSimulation()
    step_count += 1
    
    if step_count % 240 == 0:  # 1秒ごと
        print(f"  待機中... {step_count // 240}秒経過, 位置 z={pos[2]:.3f}")

print("✅ 待機完了。これからジョイントを動かします...\n")

# --- 6. 転倒後にジョイントを動かす ---
print("🦵 ジョイントを動かします（Ctrl+Cで終了）...")
print(f"   8つのジョイントを周期的に動かします\n")

step_count = 0
while True:
    # 物理演算の1ステップあたりの時間を考慮して少し待つ
    time.sleep(1.0 / 240.0)  # 240Hzでシミュレーション
    
    # 現在の位置を取得
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    
    # 周期的な動き（サイン波を使用）
    # -1.5から1.5ラジアンの範囲で動かす（URDFの制限に合わせる）
    t = step_count * 0.01  # 時間パラメータ
    
    # 8つのジョイントすべてを動かす
    # 各ジョイントに少しずつ位相をずらして動かす（見た目が分かりやすい）
    for i in range(8):
        # 各ジョイントに異なる位相を適用
        phase = i * np.pi / 4  # 各ジョイントを45度ずつずらす
        target_angle = np.sin(t + phase) * 1.0  # -1.0から1.0の範囲で振動
        
        # ジョイントを位置制御で動かす
        p.setJointMotorControl2(
            robot_id,
            i,  # ジョイントインデックス
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=5.0  # 最大速度
        )
    
    # 物理シミュレーションを1ステップ進める
    p.stepSimulation()

    step_count += 1
    
    # 定期的に状態を表示
    if step_count % 240 == 0:  # 1秒ごと
        joint_states = [p.getJointState(robot_id, i)[0] for i in range(min(4, num_joints))]
        print(f"  ステップ {step_count}: 位置 z={pos[2]:.3f}, ジョイント0-3角度: {[f'{j:.2f}' for j in joint_states]}")
