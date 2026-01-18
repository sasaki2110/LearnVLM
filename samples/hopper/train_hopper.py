import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from stable_baselines3 import PPO

# --- Step 1: 肉体(URDF)の生成（前回と同じ） ---
def create_hopper_urdf():
    urdf_content = """
    <robot name="simple_hopper">
      <link name="base">
        <visual><geometry><box size="0.2 0.2 0.2"/></geometry><material name="blue"><color rgba="0 0 1 1"/></material></visual>
        <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
        <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
      </link>
      <link name="leg">
        <visual><origin xyz="0 0 -0.25"/><geometry><cylinder length="0.5" radius="0.05"/></geometry><material name="red"><color rgba="1 0 0 1"/></material></visual>
        <collision><origin xyz="0 0 -0.25"/><geometry><cylinder length="0.5" radius="0.05"/></geometry></collision>
        <inertial><origin xyz="0 0 -0.25"/><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
      </link>
      <joint name="knee" type="revolute">
        <parent link="base"/><child link="leg"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
        <limit effort="100" lower="-1.57" upper="1.57" velocity="10"/>
      </joint>
    </robot>
    """
    with open("hopper.urdf", "w") as f:
        f.write(urdf_content)

# --- Step 2: 環境の定義 ---
class HopperEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        # 動画保存時も内部的にはDIRECT（画面なし）で動かし、ピクセルデータだけ抜く設定にします
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("hopper.urdf", [0, 0, 1.0])
        return self._get_obs(), {}

    def _get_obs(self):
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            vel, _ = p.getBaseVelocity(self.robot_id) # 速度を追加
            joint_state = p.getJointState(self.robot_id, 0)
            
            # 観測値を5つに増やす (高さ, Z軸速度, 関節角度, 関節速度, 足が地面に着いているか)
            contact = p.getContactPoints(bodyA=self.robot_id, bodyB=0)
            on_ground = 1.0 if len(contact) > 0 else 0.0
            
            return np.array([pos[2], vel[2], joint_state[0], joint_state[1], on_ground], dtype=np.float32)

    def step(self, action):
        # 1. トルクを強くする (50 -> 150)
        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=action[0] * 150.0)
        p.stepSimulation()
        
        obs = self._get_obs()
        height = obs[0]
        vz = obs[1] # Z方向の速度
        
        # 2. 報酬の設計を「躍動感重視」にする
        # 生き残っている報酬 + 高さに応じた報酬 + 上に跳ねる速度へのボーナス
        reward = 1.0  # 1ステップ生き残るごとに+1 (早く転ぶと損をする)
        reward += height * 2.0
        if vz > 0.1: # 上に向かって動いていたらボーナス
            reward += vz * 5.0
            
        # 3. 終了条件を少し厳しく（早めにリセットして次の試行へ）
        terminated = height < 0.4 
        return obs, reward, terminated, False, {}

"""
    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        joint_state = p.getJointState(self.robot_id, 0)
        return np.array([pos[2], joint_state[0], joint_state[1]], dtype=np.float32)

    def step(self, action):
        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=action[0] * 50.0)
        p.stepSimulation()
        obs = self._get_obs()
        reward = obs[0] # 高さ
        terminated = obs[0] < 0.3
        return obs, reward, terminated, False, {}

"""

# --- Step 3: 学習 ---
create_hopper_urdf()
env = HopperEnv()
model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=50000)

model_path = "ppo_hopper.zip"
model.save(model_path)
print(f"✅ 学習完了！モデルを保存しました: {model_path}")

env.close()
