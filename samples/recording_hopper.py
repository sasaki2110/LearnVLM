import pybullet as p
import pybullet_data
import numpy as np
import cv2
from stable_baselines3 import PPO

# --- 1. 定数と環境設定 ---
URDF_FILE = "hopper.urdf"
MODEL_FILE = "ppo_hopper.zip"
VIDEO_FILE = "hopper_playback.mp4"

# 録画用の設定
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF(URDF_FILE, [0, 0, 1.0])

# --- 2. モデルの読み込み ---
print(f"📦 モデル {MODEL_FILE} を読み込んでいます...")
model = PPO.load(MODEL_FILE)

# --- 3. 動画保存の設定 ---
width, height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_FILE, fourcc, 30.0, (width, height))

# --- 4. 実行と録画 ---
# 初期状態の取得
pos, _ = p.getBasePositionAndOrientation(robot_id)
joint_state = p.getJointState(robot_id, 0)
obs = np.array([pos[2], joint_state[0], joint_state[1]], dtype=np.float32)

print("🎬 再生と録画を開始します...")
for i in range(500): # 少し長めの15秒程度
    # 学習済みモデルに「次、どう動く？」と聞く
    action, _ = model.predict(obs, deterministic=True)
    
    # シミュレーション実行
    p.setJointMotorControl2(robot_id, 0, p.TORQUE_CONTROL, force=action[0] * 50.0)
    p.stepSimulation()
    
    # 次の観測値を取得
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    joint_state = p.getJointState(robot_id, 0)
    obs = np.array([pos[2], joint_state[0], joint_state[1]], dtype=np.float32)

    # カメラ設定（ロボットの動きに合わせて視点を動かす）
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[2.5, 2.5, 1.5],
        cameraTargetPosition=[pos[0], pos[1], 0.5],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100.0)
    
    # 描画
    (_, _, rgba, _, _) = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    frame = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
    out.write(frame)

    # 転倒リセット処理
    if pos[2] < 0.3:
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 1.0], [0, 0, 0, 1])

out.release()
p.disconnect()
print(f"✅ 動画を保存しました: {VIDEO_FILE}")