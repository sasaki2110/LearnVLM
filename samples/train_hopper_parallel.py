import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# --- 1. 肉体(URDF)の生成 ---
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

# --- 2. 環境の定義 ---
class HopperEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("hopper.urdf", [0, 0, 1.0])
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        joint_state = p.getJointState(self.robot_id, 0)
        return np.array([pos[2], joint_state[0], joint_state[1]], dtype=np.float32)

    def step(self, action):
        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=action[0] * 50.0)
        p.stepSimulation()
        obs = self._get_obs()
        
        # 報酬設計: 高さの維持
        reward = obs[0] 
        
        # 終了条件: 転倒（高さが0.35m以下になったら失敗）
        terminated = obs[0] < 0.35
        return obs, reward, terminated, False, {}

    def close(self):
        p.disconnect(self.client)

# --- 3. 並列化のためのヘルパー関数 ---
def make_env(rank, seed=0):
    def _init():
        env = HopperEnv()
        # 各環境でシード値をずらして多様なデータを集める
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# --- 4. メイン学習処理 ---
if __name__ == "__main__":
    # URDF作成
    create_hopper_urdf()

    # 並列数の設定 (CPUのコア数に合わせて調整してください)
    num_cpu = 4 
    print(f"🚀 {num_cpu} 並列で学習を開始します...")

    # ベクトル化環境の作成
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # モデルの定義
    # tensorboard_log を指定しておくと学習過程を後でグラフで見ることができます
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", # GPUが使える場合はcuda
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
    )

    # 100万ステップ学習 (1時間程度の目安)
    TOTAL_TIMESTEPS = 1000000
    print(f"⌛ {TOTAL_TIMESTEPS} ステップの学習を実行中...")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # モデルの保存
    model_path = "ppo_hopper_parallel.zip"
    model.save(model_path)
    print(f"✅ 学習完了！モデルを保存しました: {model_path}")

    env.close()