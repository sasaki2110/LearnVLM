import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. TensorBoardにカスタム報酬を送るためのコールバック ---
class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        # info辞書に入れた custom/ データを抽出してログに記録
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            for key, value in info.items():
                if key.startswith("custom/"):
                    self.logger.record(key, value)
        return True

# --- 2. 学習環境の定義 ---
class Vision60TrotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        # 観測: 高さ(1), Z速度(1), 姿勢(3), ジョイント(12), フェーズ(1) = 18次元
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.step_count = 0

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # 初期姿勢を少し高くし、わずかにランダムな傾きを与えて「どや顔」を防止
        rand_orn = p.getQuaternionFromEuler([np.random.uniform(-0.05, 0.05), 
                                             np.random.uniform(-0.05, 0.05), 0])
        self.robot_id = p.loadURDF("quadruped/vision60.urdf", [0, 0, 0.55], rand_orn)
        
        self.step_count = 0
        for i, j_idx in enumerate(self.joint_indices):
            init_pos = 1.2 if i in [2, 5, 8, 11] else 0.0
            p.resetJointState(self.robot_id, j_idx, init_pos)
            
        return self._get_obs(), {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, _ = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        phase = np.sin(2 * np.pi * 1.5 * (self.step_count * 0.01)) 
        return np.array([pos[2], vel[2]] + list(euler) + joint_angles + [phase], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        t = self.step_count * 0.01
        phase_a = np.sin(2 * np.pi * 1.5 * t)
        phase_b = -phase_a

        # 関節制御
        for i, j_idx in enumerate(self.joint_indices):
            target_pos = 1.2 
            if i in [2, 8]: target_pos += phase_a * 0.4   # FR, RL
            elif i in [5, 11]: target_pos += phase_b * 0.4 # FL, RR
            
            target_pos += action[i] * 0.2
            target_pos = np.clip(target_pos, 0.1, 3.0)
            p.setJointMotorControl2(self.robot_id, j_idx, p.POSITION_CONTROL, targetPosition=target_pos, force=180.0)

        p.stepSimulation()
        obs = self._get_obs()
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # 終了判定（厳格化）
        is_falling = pos[2] < 0.35 or abs(euler[0]) > 0.4 or abs(euler[1]) > 0.5
        terminated = False
        reward = 1.0 
        
        if self.step_count > 20 and is_falling:
            terminated = True
            reward = -100.0
        
        # 報酬計算とログ用変数の準備
        knee_fl = p.getJointState(self.robot_id, self.joint_indices[5])[0]
        knee_fr = p.getJointState(self.robot_id, self.joint_indices[2])[0]
        knee_diff = abs(knee_fl - knee_fr)
        
        if not terminated:
            reward -= abs(pos[2] - 0.45) * 15.0
            reward -= abs(euler[0]) * 30.0 # 横揺れペナルティ
            
            # 左右対称禁止報酬：左右の差が大きいほど加点
            reward += (knee_diff - 0.2) * 10.0 
            
            vel, _ = p.getBaseVelocity(self.robot_id)
            reward += vel[0] * 50.0

        # TensorBoardで見たい数値をinfoに入れる
        info = {
            "custom/knee_diff": knee_diff,
            "custom/height": pos[2],
            "custom/roll": abs(euler[0])
        }

        return obs, reward, terminated, False, info

# --- 3. 学習実行メイン ---
if __name__ == "__main__":
    # GPUがない場合は単一、ある場合は複数プロセスを推奨
    env = Vision60TrotEnv() 
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu", # GPUがない場合は明示的にCPU
        tensorboard_log="./ppo_vision_logs/"
    )
    
    print("🐾 学習を開始します。TensorBoardを別ターミナルで起動して待機してください。")
    callback = TensorboardCallback()
    model.learn(total_timesteps=500000, callback=callback)
    model.save("ppo_vision60_trot_final")