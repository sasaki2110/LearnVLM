# VLM 学習環境

## ステップ１：環境構築

```bash
# 1. 基本的な依存関係のインストール
apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libglib2.0-0 \
    python3-pil

# 不足分のインストール
apt install python3.12-venv
apt install -y curl

# 2. 仮想環境を作成（名前は .venv としています）
python3 -m venv .venv

# 3. 仮想環境を有効化
source .venv/bin/activate

# 4. 仮想環境内でインストール（これならエラーが出ません）
# PyTorch (CUDA 12.1対応版)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# VLM推論に必要なライブラリ
pip install transformers timm einops pillow
```

## ステップ 2：VLM（Moondream2）を実際に動かす

では、さっそく「画像を見て、内容を言葉にする」コードを動かしてみましょう。

### 1. 必要なファイルの準備

まずは、VLMに読み込ませる画像が必要です。ネット上の画像をダウンロードするか、手元の画像をコンテナにコピーしてください。

```bash
# ネット上のテスト画像をダウンロードする場合
curl -o test_image.jpg https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg

```

### 2. 推論スクリプトの作成

`~/LearnVLM/vlm_test.py` を作成して、以下のコードを貼り付けてください。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# 1. モデルとトークナイザーの準備
# RTX 3060 Ti (8GB) なら、このモデルは余裕で動きます
model_id = "vikhyatk/moondream2"
revision = "2024-08-05"

print("🚀 モデルをロード中...（初回は数分かかります）")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    revision=revision,
    torch_dtype=torch.float16, # 半精度でメモリ節約
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model.eval() # 推論モードに設定

# 2. 画像の読み込みとエンコード
image_path = "test_image.jpg"
image = Image.open(image_path)
print(f"📸 画像を読み込みました: {image_path}")

# VLMが画像を理解するための「視覚特徴」を抽出
enc_image = model.encode_image(image)

# 3. 質問して回答を得る
question = "Describe this image in one sentence."
print(f"❓ 質問: {question}")

answer = model.answer_question(enc_image, question, tokenizer)
print(f"💡 回答: {answer}")

# ロボットアームの操作を意識した質問例
# question = "What objects are on the table?"
# question = "Where is the red object located in the image?"

```

### 3. 実行

```bash
python vlm_test.py

```

初回の実行時にはモデルのダウンロード（約 3GB 程度）が始まります。ダウンロードが終われば、RTX 3060 Ti のパワーで爆速（1秒以内）で回答が返ってくるはずです。

---

## このステップで学べること

このコードが動くと、以下のことが実感できるはずです。

* **物理的な画像が「テキスト」になる:** これまで PyBullet で「座標」として扱っていた世界が、「言葉」で記述できる世界に繋がります。
* **ゼロショット認識:** このモデルに「コップはどこ？」「アームは動いている？」と聞けば、事前の学習なしにその場で答えてくれます。

これができれば、次のステップはいよいよ **「PyBullet で生成した画像を、この VLM に見せて、LangGraph の指示にフィードバックする」** という、真のフィジカル AI のループに突入できます。

## ステップ３：追加の環境設定（PyBullet）

```bash
# PyBullet本体のインストール
pip install pybullet

# 画像処理をスムーズにするための追加ライブラリ
pip install numpy matplotlib
```

## GUI で動かす場合

### OpenGL関連のインストール
```bash
apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl-mesa0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxfixes3 \
    libxi6 \
    libxinerama1
```

### 環境変数定義
```bash
export DISPLAY=host.docker.internal:0.0
```

## ステップ４：LangGraphの設定


```bash
# 仮想化環境を作成
python3 -m venv .venv

# 仮想環境を有効化
source .venv/bin/activate

# pipをアップグレード
pip install --upgrade pip

# requirements.txtからインストール GPU有り版。　CPUしかない環境は後述。
pip install -r requirements_langgraph.txt

# CPU しかない環境の依存関係をインストール
pip install -r requirements_cpu.txt

# 動画記録用
pip install opencv-python


```

## ステップ５：機械学習環境の設定

```bash
pip install gymnasium stable-baselines3 shimmy pybullet
pip install opencv-python
```

## ステップ６：機械学習環境の確認

### インストール

```bash
pip install tensorboard
```

### ログ出力
```python
# GPUがなくても device="cpu" と指定すれば動きます
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    device="cpu", 
    tensorboard_log="./ppo_vision_logs/"  # ← ログの保存先を指定
)
```

### 起動（別ターミナルで、ディレクトリ階層を合わせて）

```bash
tensorboard --logdir ./ppo_vision_logs/
```

###ブラウザで確認。

ブラウザで確認: http://localhost:6006/ を開きます。

### CUSTOM ログ採取

```python
    def step(self, action):
        # ... (前述の制御や報酬計算コード) ...

        # --- ログ出力用の情報を info 辞書に入れる ---
        # ここに書いた数値が TensorBoard で個別のグラフになります
        info = {
            "custom/knee_diff": knee_diff_front,  # 左右の足のズレ
            "custom/height": pos[2],              # 高さ
            "custom/roll": abs(euler[0]),         # 横揺れ
            "custom/pitch": abs(euler[1])         # 縦揺れ
        }

        return obs, reward, terminated, False, info # infoを返す
```

```python
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        # info 辞書の中身を TensorBoard に書き出す
        if "custom/knee_diff" in self.locals["infos"][0]:
            for key in self.locals["infos"][0].keys():
                if key.startswith("custom/"):
                    self.logger.record(key, self.locals["infos"][0][key])
        return True

# --- 学習開始部分 ---
callback = TensorboardCallback()
model.learn(total_timesteps=500000, callback=callback) # callbackを追加
```
