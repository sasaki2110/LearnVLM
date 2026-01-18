import pybullet as p
import pybullet_data
import numpy as np
import time  # リアルタイム再生の速度調整用
from stable_baselines3 import PPO

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
# もしこれでエラーが出る場合は、WSL2のGUI設定に課題があります
print("🚀 PyBulletをGUIモードで起動します...")
device_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# URDFファイルが存在しない場合は生成
import os
if not os.path.exists("quad.urdf"):
    create_quadruped_urdf()
else:
    print("✅ quad.urdf が見つかりました")

# 四足ロボットをロード（学習時と同じ高さから開始）
robot_id = p.loadURDF("quad.urdf", [0, 0, 0.3])
print("✅ 四足ロボットをロードしました")

# ジョイント情報を表示
num_joints = p.getNumJoints(robot_id)
print(f"📊 ジョイント数: {num_joints}")
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    print(f"  ジョイント {i}: {joint_name}")

# --- 3. モデルの読み込み ---
model_path = "ppo_quad.zip"
try:
    model = PPO.load(model_path)
    print(f"✅ モデル '{model_path}' を読み込みました")
except FileNotFoundError:
    print(f"❌ エラー: モデルファイル '{model_path}' が見つかりません")
    print(f"   先に train_quad.py を実行してモデルを学習してください")
    exit(1)

# --- 4. 初期観測値を取得 ---
def get_obs():
    """観測値を取得（学習時と同じ形式）"""
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    vel, _ = p.getBaseVelocity(robot_id)
    joint_states = [p.getJointState(robot_id, i)[0] for i in range(8)]
    return np.array([pos[2], vel[2]] + joint_states, dtype=np.float32)

obs = get_obs()

# デフォルトの速度制御（ブレーキ）を無効化
for i in range(8):
    p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)

print("\n📺 GUIで再生を開始します（Ctrl+Cで終了）...")
print("   四足ロボットの跳躍動作を表示します\n")
input("⏸️  Enterキーを押すと再生を開始します...")

# --- 5. 実行ループ ---
while True:
    # 物理演算の1ステップあたりの時間を考慮して少し待つ（これがないと超高速で終わります）
    time.sleep(1.0 / 240.0)
    
    # アクションを予測
    action, _ = model.predict(obs, deterministic=True)
    
    # 8つの関節すべてにアクションを適用（学習時と同じ）
    for i in range(8):
        p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=action[i] * 100.0)
    
    p.stepSimulation()
    
    # 次の状態取得
    obs = get_obs()
    
    # 転倒リセット（胴体が極端に低くなったら）
    if obs[0] < 0.15:
        print("⚠️  転倒を検出。リセットします...")
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.3], [0, 0, 0, 1])
        # ジョイントもリセット
        for i in range(8):
            p.resetJointState(robot_id, i, 0, 0)
        obs = get_obs()
        # デフォルトの速度制御を再度無効化
        for i in range(8):
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
