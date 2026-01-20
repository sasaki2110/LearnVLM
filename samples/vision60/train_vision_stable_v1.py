import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. TensorBoardログ出力用コールバック ---
class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
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
        
        # 1. スポーン位置を低く設定（0.3m）
        spawn_height = 0.3
        self.robot_id = p.loadURDF("quadruped/vision60.urdf", [0, 0, spawn_height])
        
        self.step_count = 0
        
        # 2. 初期姿勢：膝を1.0曲げ、Abduction(肩)を「ハの字(0.2)」に開く
        knee_angle = 1.0
        abd_angle = 0.2
        
        # Abductionジョイントの符号を個別に設定（check_vision.pyと同じ設定）
        # i=0, j_idx=0: 左前（FL）
        # i=3, j_idx=4: 右前（FR）
        # i=6, j_idx=8: 左後ろ（RL）
        # i=9, j_idx=12: 右後ろ（RR）
        abd_signs = {
            0: 1.0,   # 左前（FL）: +1.0でプラス、-1.0でマイナス
            3: 1.0,   # 右前（FR）: +1.0でプラス、-1.0でマイナス
            6: -1.0,  # 左後ろ（RL）: +1.0でプラス、-1.0でマイナス
            9: -1.0,  # 右後ろ（RR）: +1.0でプラス、-1.0でマイナス
        }
        
        for i, j_idx in enumerate(self.joint_indices):
            if i in [0, 3, 6, 9]: # Abductionジョイント (ハの字)
                init_pos = abd_angle * abd_signs[i]
            elif i in [2, 5, 8, 11]: # Knee
                init_pos = knee_angle
            else: # Hip
                init_pos = 0.0
            
            p.resetJointState(self.robot_id, j_idx, init_pos)
            # 初期状態で崩れないようモーターを保持
            p.setJointMotorControl2(self.robot_id, j_idx, p.POSITION_CONTROL, 
                                    targetPosition=init_pos, force=150.0)

        # 3. 安定待ちフェーズ（100ステップ）
        # これにより「着地の衝撃」が収まってから学習がスタートします
        for _ in range(100):
            p.stepSimulation()
        
        return self._get_obs(), {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, _ = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        # トロット用の基準フェーズ（1.5Hz）
        phase = np.sin(2 * np.pi * 1.5 * (self.step_count * 0.01)) 
        return np.array([pos[2], vel[2]] + list(euler) + joint_angles + [phase], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        t = self.step_count * 0.01
        phase_a = np.sin(2 * np.pi * 1.5 * t)
        phase_b = -phase_a

        # 関節制御（しなやか設定）
        for i, j_idx in enumerate(self.joint_indices):
            # 基本姿勢の維持
            if i in [0, 3, 6, 9]: target_pos = 0.2 if i in [0, 6] else -0.2 # ハの字保持
            elif i in [2, 8]: target_pos = 1.0 + phase_a * 0.4 # FR, RL knee
            elif i in [5, 11]: target_pos = 1.0 + phase_b * 0.4 # FL, RR knee
            else: target_pos = 0.0
            
            # AIのアクションを加算
            target_pos += action[i] * 0.2
            
            # positionGainを下げて「柔らかく」、velocityGainを上げて「粘り」を出す
            p.setJointMotorControl2(
                self.robot_id, j_idx, p.POSITION_CONTROL, 
                targetPosition=target_pos, 
                force=100.0, 
                positionGain=0.05, 
                velocityGain=1.5
            )

        p.stepSimulation()
        obs = self._get_obs()
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # 判定
        # 1.5秒間（150ステップ）は転倒判定を行わない「執行猶予」
        is_falling = (pos[2] < 0.28 or abs(euler[0]) > 0.6 or abs(euler[1]) > 0.6)
        
        terminated = False
        reward = 1.0 # 生存報酬（固定値）
        
        if self.step_count > 150 and is_falling:
            terminated = True
            reward = -100.0
        
        # 補助報酬
        knee_fl = p.getJointState(self.robot_id, self.joint_indices[5])[0]
        knee_fr = p.getJointState(self.robot_id, self.joint_indices[2])[0]
        knee_diff = abs(knee_fl - knee_fr)
        
        if not terminated:
            # 高さ維持（重みを5.0に軽減）
            reward -= abs(pos[2] - 0.4) * 5.0
            # 左右対称禁止報酬
            reward += (knee_diff - 0.2) * 10.0 
            # 前進報酬
            vel, _ = p.getBaseVelocity(self.robot_id)
            reward += vel[0] * 30.0

        info = {
            "custom/knee_diff": knee_diff,
            "custom/height": pos[2],
            "custom/roll": abs(euler[0])
        }

        return obs, reward, terminated, False, info

# --- 3. 学習実行 ---
if __name__ == "__main__":
    # 既存のログがある場合は削除するかフォルダ名を変えることを推奨
    env = Vision60TrotEnv() 
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu",
        tensorboard_log="./ppo_vision_logs/"
    )
    
    callback = TensorboardCallback()
    print("🐾 安定化リセット＆しなやか制御で学習を開始します...")
    model.learn(total_timesteps=500000, callback=callback)
    model.save("ppo_vision60_stable_v1")