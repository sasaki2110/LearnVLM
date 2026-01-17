import pybullet as p
import pybullet_data
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

# --- 1. VLMの準備 ---
model_id = "vikhyatk/moondream2"
print("🚀 VLMをロード中...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# --- 2. PyBulletの設定 (GUIなし・EGL有効) ---
print("物理シミュレータを起動中...")
p.connect(p.DIRECT) # 画面を表示しないモード
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 床と物体（例：アヒルちゃんやブロック）を配置
p.loadURDF("plane.urdf")
obj_id = p.loadURDF("duck_vhacd.urdf", basePosition=[0.5, 0.2, 0.0]) # 少しズラして配置

# --- 3. カメラ撮影 ---
print("📸 カメラで撮影中...")
width, height = 640, 480

# 視点（カメラの位置）を直接指定
camera_eye = [1.0, 1.0, 1.0]    # カメラ本体の座標 (x, y, z)
camera_target = [0.5, 0.2, 0.0] # カメラが向く先（アヒルを置いた座標）
camera_up = [0, 0, 1]           # 上方向のベクトル

view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0)

# 画像取得
_, _, rgb_img, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)

# NumPy配列をPIL画像に変換
rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3] # RGBA -> RGB
raw_image = Image.fromarray(rgb_array.astype('uint8'))
raw_image.save("sim_capture.jpg") # 確認用に保存

# --- 4. VLMによる解析 ---
print("🧠 VLMで物体を検出中...")
enc_image = model.encode_image(raw_image)

# 先ほど成功したプロンプトを使用
question = "Point out the duck with a bounding box."
answer = model.answer_question(enc_image, question, tokenizer)

print("-" * 30)
print(f"💡 VLMの回答: {answer}")
print("-" * 30)

p.disconnect()