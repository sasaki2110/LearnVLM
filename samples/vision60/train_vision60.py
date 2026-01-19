import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# --- 1. 環境の定義 ---
class Vision60PositionEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # AIの出力: 12個の関節の「目標角度」 (-1 ~ 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        # 観測: 高さ(1), Z速度(1), 姿勢(3), ジョイント角(12) = 17次元
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        self.joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # 0.5mの高さから生成
        self.robot_id = p.loadURDF("quadruped/vision60.urdf", [0, 0, 0.5])
        
        # 初期姿勢: 少し膝を曲げておくと立ち上がりやすい
        for i, j_idx in enumerate(self.joint_indices):
            # 膝(Knee)はIndex 2, 5, 8, 11 (リスト内では 2, 5, 8, 11番目)
            if i in [2, 5, 8, 11]:
                p.resetJointState(self.robot_id, j_idx, 1.0) 
            
        return self._get_obs(), {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, _ = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        return np.array([pos[2], vel[2]] + list(euler) + joint_angles, dtype=np.float32)

    def step(self, action):
        # AIの出力を実際の関節角度(ラジアン)にスケーリング
        for i, j_idx in enumerate(self.joint_indices):
            if i in [0, 3, 6, 9]: # Abduction (±0.43 rad)
                target_pos = action[i] * 0.43
                force = 300.0
            elif i in [1, 4, 7, 10]: # Hip (±3.14 rad)
                target_pos = action[i] * 3.14
                force = 80.0
            else: # Knee (0 ~ 3.14 rad)
                # action[-1, 1] -> [0, 3.14]
                target_pos = (action[i] + 1) * 1.57
                force = 80.0

            p.setJointMotorControl2(
                self.robot_id, j_idx, p.POSITION_CONTROL, 
                targetPosition=target_pos, force=force
            )
        
        p.stepSimulation()
        obs = self._get_obs()
        
        height = obs[0]
        roll, pitch = obs[2], obs[3]
        base_vel, _ = p.getBaseVelocity(self.robot_id)
        
        """
        # --- 報酬設計：より厳格な姿勢制御 ---
        reward = 1.0  # 基本生存報酬
        reward -= abs(height - 0.5) * 10.0  # 高さがずれると大幅減点
        reward -= (abs(roll) + abs(pitch)) * 15.0  # 傾くと超大幅減点（スパルタ）
        reward += base_vel[0] * 20.0  # 前進への強いインセンティブ
        # --- 報酬設計：積極的歩行プラン ---
        reward = 1.0  # 生存報酬

        # 前進報酬を大幅に強化 (これが歩くエネルギーになる)
        forward_vel = base_vel[0]
        reward += forward_vel * 100.0  # 5倍に強化！

        # 姿勢と高さの維持（これは「どや顔」の元なので、少し係数を下げるか、許容範囲を広げる）
        reward -= abs(height - 0.5) * 5.0
        reward -= (abs(roll) + abs(pitch)) * 10.0

        # 【隠し味】横方向へのフラつきや、その場での回転を抑制
        reward -= abs(base_vel[1]) * 10.0  # 横歩き禁止
        reward -= abs(obs[4]) * 5.0        # Yaw（回転）禁止
        
        # 終了判定：高さが低い、または大きく傾いたら即終了
        terminated = height < 0.3 or abs(roll) > 0.5 or abs(pitch) > 0.5
        """

        # --- 終了判定の強化：つんのめり禁止 ---
        # pitch < -0.3 は前につんのめっている状態
        terminated = height < 0.35 or abs(roll) > 0.4 or pitch < -0.3 or pitch > 0.4
        
        # --- 報酬の再設計：頭を擦っても得させない ---
        reward = 1.0 
        
        # 前進報酬：ただし、姿勢がまともな時だけボーナスを出す
        if height > 0.4 and abs(pitch) < 0.2:
            reward += base_vel[0] * 150.0  # 良い姿勢での前進は超高得点！
        else:
            reward += base_vel[0] * 10.0   # 悪い姿勢（頭擦り）は低得点
            
        # 左右へのフラつき制限
        reward -= abs(base_vel[1]) * 5.0        
        
        return obs, reward, terminated, False, {}

    def close(self):
        p.disconnect(self.client)

# --- 2. 学習の実行 ---
def make_env(rank):
    def _init(): return Vision60PositionEnv()
    return _init

if __name__ == "__main__":
    num_cpu = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # ネットワーク構造を少し深くして、複雑な姿勢を学べるようにする
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cuda")
    
    print("🐕 Vision60 位置制御(Position Control)での学習を開始します...")
    model.learn(total_timesteps=500000)
    model.save("ppo_vision60_position_step")
    print("✅ 学習完了。 ppo_vision60_position_step.zip を保存しました。")