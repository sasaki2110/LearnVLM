import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# 1. モデルとトークナイザーの準備
# RTX 3060 Ti (8GB) なら、このモデルは余裕で動きます
model_id = "vikhyatk/moondream2"
#revision = "2024-08-05"

print("🚀 モデルをロード中...（初回は数分かかります）")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    #revision=revision,
    torch_dtype=torch.float16, # 半精度でメモリ節約
).to("cuda")

#tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval() # 推論モードに設定

# 2. 画像の読み込みとエンコード
image_path = "test_image.jpg"
image = Image.open(image_path)
print(f"📸 画像を読み込みました: {image_path}")

# VLMが画像を理解するための「視覚特徴」を抽出
enc_image = model.encode_image(image)

# 3. 質問して回答を得る
question = "Describe this image in one sentence."
#question = "Locate the bread roll in the image."
# 「物体検出（Object Detection）」を明示的に指示します
#question = "Detect the bread roll in the image. Respond with a JSON object containing the coordinates."

#question = "Point out the bread roll with a bounding box."
print(f"❓ 質問: {question}")

answer = model.answer_question(enc_image, question, tokenizer)
print(f"💡 回答: {answer}")

# ロボットアームの操作を意識した質問例
# question = "What objects are on the table?"
# question = "Where is the red object located in the image?"

