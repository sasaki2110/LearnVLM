import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# --- 1. 四足L字ロボット(URDF)の生成 ---
def create_quadruped_urdf():
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
    with open("quad.urdf", "w") as f: f.write(urdf_content)
    
# --- 2. 環境の定義 ---
class QuadrupedEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # アクション: 8つの関節のトルク (-1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        # 観測: 胴体の高さ, Z速度, 8つの関節角度 (計10個)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        # ちょうどいい高さ(0.3m)から落とす
        self.robot_id = p.loadURDF("quad.urdf", [0, 0, 0.3])
        # 最初は少し待って安定させる（物理演算を空回し）
        for _ in range(20): p.stepSimulation()
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        vel, _ = p.getBaseVelocity(self.robot_id)
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(8)]
        return np.array([pos[2], vel[2]] + joint_states, dtype=np.float32)

    def step(self, action):
        # 8つの関節すべてにアクションを適用
        for i in range(8):
            p.setJointMotorControl2(self.robot_id, i, p.TORQUE_CONTROL, force=action[i] * 100.0)
        
        p.stepSimulation()
        obs = self._get_obs()
        height, vz = obs[0], obs[1]
        
        # 報酬: 「飛ぶ」＝高さへの報酬 ＋ 上向きの速度への大きなボーナス
        reward = height * 2.0
        if vz > 0.2: reward += vz * 10.0 # ジャンプへの強い意欲
        
        # 胴体が極端に低くなったら（転倒したら）終了
        terminated = height < 0.15
        return obs, reward, terminated, False, {}

# --- 3. 学習（並列実行対応） ---
def make_env(rank):
    def _init(): return QuadrupedEnv()
    return _init

if __name__ == "__main__":
    create_quadruped_urdf()
    num_cpu = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    print("🚀 四足ロボットの跳躍学習を開始します...")
    model.learn(total_timesteps=500000) # まずは30分〜1時間程度の50万回
    model.save("ppo_quad.zip")
    print("✅ 学習完了！ ppo_quad.zip を保存しました。")