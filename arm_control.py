import pybullet as p
import pybullet_data
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

# --- 1. PyBullet設定 ---
#p.connect(p.DIRECT) 
p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 地面とアーム、アヒルのロード
p.loadURDF("plane.urdf")
arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
duck_pos = [0.5, 0.2, 0.0]
obj_id = p.loadURDF("duck_vhacd.urdf", basePosition=duck_pos)

# アームの関節数を確認
num_joints = p.getNumJoints(arm_id)
ee_index = num_joints - 1  # 手先のリンクインデックス

# --- 2. 視覚認識（前回と同じ） ---
def get_visual_observation():
    width, height = 640, 480
    camera_eye = [1.0, 1.0, 1.0]
    camera_target = [0.5, 0.2, 0.0]
    view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, float(width)/height, 0.1, 100.0)
    _, _, rgb_img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    return Image.fromarray(rgb_array.astype('uint8'))

# --- 3. 動作実行関数 ---
def move_arm_to(target_pos):
    print(f"🦾 ターゲット {target_pos} へ移動中...")
    # 逆運動学で各関節の目標角度を計算
    joint_poses = p.calculateInverseKinematics(arm_id, ee_index, target_pos)
    
    # 計算された角度を各関節に適用
    for i in range(len(joint_poses)):
        p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_poses[i])
    
    # シミュレーションを少し進めて動きを反映させる
    for _ in range(100):
        p.stepSimulation()

# --- 実行ループ ---
# 本来はここでVLMから座標を取得しますが、まずは物理的な「動作」を確認します
target_location = [0.5, 0.2, 0.2] # アヒルの少し上空
move_arm_to(target_location)

# 到着後の手先の座標を確認
current_ee_pos = p.getLinkState(arm_id, ee_index)[0]
print(f"📍 到着した手先の座標: {current_ee_pos}")

print("\nEnterキーを押すと終了します...")

try:
    input()
except KeyboardInterrupt:
    pass


p.disconnect()