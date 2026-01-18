import pybullet as p
import pybullet_data
import numpy as np
import time  # リアルタイム再生の速度調整用
from stable_baselines3 import PPO

# --- 1. GUIモードで接続 ---
# もしこれでエラーが出る場合は、WSL2のGUI設定に課題があります
device_id = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("hopper.urdf", [0, 0, 1.0])

# --- 2. モデルの読み込み --- 
model = PPO.load("ppo_hopper.zip")
#model = PPO.load("ppo_hopper_parallel.zip")

# --- 3. 実行 ---
"""
pos, _ = p.getBasePositionAndOrientation(robot_id)
joint_state = p.getJointState(robot_id, 0)
obs = np.array([pos[2], joint_state[0], joint_state[1]], dtype=np.float32)
"""
pos, _ = p.getBasePositionAndOrientation(robot_id)
vel, _ = p.getBaseVelocity(robot_id) # 速度を追加
joint_state = p.getJointState(robot_id, 0)

# 観測値を5つに増やす (高さ, Z軸速度, 関節角度, 関節速度, 足が地面に着いているか)
contact = p.getContactPoints(bodyA=robot_id, bodyB=0)
on_ground = 1.0 if len(contact) > 0 else 0.0

obs = np.array([pos[2], vel[2], joint_state[0], joint_state[1], on_ground], dtype=np.float32)

# デフォルトの速度制御（ブレーキ）を無効化
p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, force=0)

print("📺 GUIで再生を開始します（Ctrl+Cで終了）...")

while True:
    # 物理演算の1ステップあたりの時間を考慮して少し待つ（これがないと超高速で終わります）
    time.sleep(1./240.) 
    
    action, _ = model.predict(obs, deterministic=True)
    #p.setJointMotorControl2(robot_id, 0, p.TORQUE_CONTROL, force=action[0] * 50.0)
    p.setJointMotorControl2(robot_id, 0, p.TORQUE_CONTROL, force=action[0] * 150.0)
    p.stepSimulation()
    
    # 次の状態取得
    """
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    joint_state = p.getJointState(robot_id, 0)
    obs = np.array([pos[2], joint_state[0], joint_state[1]], dtype=np.float32)
    """
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    vel, _ = p.getBaseVelocity(robot_id) # 速度を追加
    joint_state = p.getJointState(robot_id, 0)

    # 観測値を5つに増やす (高さ, Z軸速度, 関節角度, 関節速度, 足が地面に着いているか)
    contact = p.getContactPoints(bodyA=robot_id, bodyB=0)
    on_ground = 1.0 if len(contact) > 0 else 0.0

    obs = np.array([pos[2], vel[2], joint_state[0], joint_state[1], on_ground], dtype=np.float32)

    # 転倒リセット
    if pos[2] < 0.3:
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 1.0], [0, 0, 0, 1])
        p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, force=0)