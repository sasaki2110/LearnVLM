# p01_ex_arm - VLMロボットブリッジエージェント

このエージェントは、VLM（Visual Language Model）を使用してロボットアームを制御するシンプルなエージェントです。
`vlm_robot_bridge.py`をベースにしたツール関数を提供します。

## 構造

```
p01_ex_arm/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py          # グラフ定義（start -> vlm_robot_bridge -> end）
│   └── utils/
│       ├── __init__.py
│       ├── state.py      # 状態定義（messagesのみ）
│       ├── pybullet_env.py  # PyBullet環境管理
│       ├── tools.py      # vlm_robot_bridge()ツール
│       ├── nodes.py      # ノード関数
│       └── logging_config.py  # ロギング設定
├── tests/
│   ├── __init__.py
│   └── test_invoke.py
├── setup.py
├── langgraph.json
└── README.md
```

## 機能

このエージェントは、以下の処理を行います：

1. **VLMロボットブリッジ**: VLMを使用してアヒルを検出し、ロボットアームで段階的に接近します
   - ステップ1: VLMで目標位置を決定
   - ステップ2: 目標位置の10cm上まで段階的に移動（10段階）
   - ステップ3: 真上に移動するまでループ
   - ステップ4: 少しずつ下がる（10段階）

## セットアップ

### 1. 依存関係のインストール

プロジェクトルート（`/root/LearnVLM`）で以下のコマンドを実行：

```bash
source .venv/bin/activate
pip install -r requirements_langgraph.txt
pip install pybullet numpy torch transformers pillow
```

### 2. パッケージのインストール（重要！）

`p01_ex_arm`ディレクトリで、パッケージをインストールします：

```bash
cd /root/LearnVLM/agents/p01_ex_arm
pip install -e .
```

### 3. 環境変数の設定

`.env` ファイルに以下を設定（プロジェクトルート `/root/LearnVLM/.env`）：

```
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key_here
```

### 4. LangGraph Studioで実行

```bash
cd /root/LearnVLM/agents/p01_ex_arm
langgraph dev
```

サーバーが起動すると、以下のURLでアクセスできます：
- **APIエンドポイント**: `http://127.0.0.1:2024`
- **Studio UI**: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

## 使用方法

エージェントにメッセージを送信すると、VLMロボットブリッジが実行されます：

```
"VLMロボットブリッジを実行してください"
```

GUIモードを使用する場合は：

```
"VLMロボットブリッジをGUIモードで実行してください"
```

## テスト

```bash
cd /root/LearnVLM/agents/p01_ex_arm
python -m pytest tests/test_invoke.py -v
```
