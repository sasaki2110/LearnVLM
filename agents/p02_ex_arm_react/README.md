# p02_ex_arm_react - ReAct型フィジカルAIエージェント

ReAct（Reasoning + Acting）パターンを使用したフィジカルAIエージェントです。
LangGraphを使用して、思考→行動→観察のループを実現します。

## 構造

```
p02_ex_arm_react/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py          # グラフ定義
│   └── utils/
│       ├── __init__.py
│       ├── state.py       # 状態定義
│       ├── nodes.py        # ノード実装（Agent, ToolNode）
│       ├── tools.py        # ツール定義
│       ├── pybullet_env.py # PyBullet環境
│       └── logging_config.py
├── tests/
│   ├── __init__.py
│   └── test_invoke.py
├── setup.py
└── README.md
```

## 機能

### ツール定義

1. **find_object(target_name: str)**
   - VLM（moondream2）を使用して物体を検出
   - 2D座標を3D空間座標に変換
   - 戻り値: 物体の [X, Y, Z] 座標

2. **move_arm(x, y, z)**
   - 指定された [X, Y, Z] へアームを移動
   - 逆運動学（p.calculateInverseKinematics）を使用
   - 戻り値: 移動後の現在地

3. **grasp_object()**
   - 手先の最も近くにある物体をアームに固定
   - p.createConstraint を使用

4. **release_object()**
   - 固定を解除
   - p.removeConstraint を使用

### PyBullet環境

- KUKA IIWAアームを [0, 0, 0] に固定
- アヒル（duck_vhacd.urdf）をランダムな位置に配置
- トレイ（tray.urdf）を別のランダムな位置に配置
- 500ステップの安定化処理

### ReActエージェント

LangGraphの標準的なReActパターンを使用：

1. **Agentノード（思考）**
   - LLMが状況を見て、「思考（Thought）」と「行動（Action）」を出力
   - 使えるツール: find_object, move_arm, grasp_object, release_object
   - 現在の状況（アーム位置、掴んでいる物体、前回のツール結果）を考慮

2. **ToolNode（実行）**
   - Agentノードが指示したツールを実行
   - 結果（Observation）を返す

3. **条件分岐**
   - ツールが実行されたら、再びAgentノードに戻る
   - LLMが満足（「目標達成！」と発言）したら終了

## セットアップ

### 1. 依存関係のインストール

プロジェクトルート（`/root/LearnVLM`）で：

```bash
source .venv/bin/activate
pip install -r requirements_langgraph.txt
```

### 2. パッケージのインストール（不要）

**このエージェントは`pip install -e .`なしで動作します。**
テストファイルとエージェントコードが自動的にパスを設定するため、`pip install -e .`は不要です。

**既存の`*.egg-info`ディレクトリがある場合**:
```bash
# 不要なegg-infoディレクトリを削除（オプション）
cd /root/LearnVLM/agents/p02_ex_arm_react
rm -rf *.egg-info
```

**注意**: もし`pip install -e .`を使用する場合は、他のエージェント（`p00_sample`、`p01_ex_arm`など）を先にアンインストールしてください：

```bash
# 他のエージェントをアンインストール（競合を避けるため）
pip uninstall -y p00_sample p01_ex_arm

# このエージェントをインストール
cd /root/LearnVLM/agents/p02_ex_arm_react
pip install -e .
```

### 3. 環境変数の設定

`.env` ファイルに以下を設定（プロジェクトルート `/root/LearnVLM/.env`）：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## 使用方法

### テスト実行

```bash
cd /root/LearnVLM/agents/p02_ex_arm_react
python tests/test_invoke.py
```

### 動画記録付きで実行

シミュレーションの動きを動画として記録する場合：

```bash
cd /root/LearnVLM/agents/p02_ex_arm_react
RECORD_VIDEO=true VIDEO_FILENAME=react_agent_simulation.mp4 python tests/test_invoke.py
```

または：

```bash
export RECORD_VIDEO=true
export VIDEO_FILENAME=react_agent_simulation.mp4
python tests/test_invoke.py
```

動画は実行後に `react_agent_simulation.mp4`（または指定したファイル名）として保存されます。

**注意**: 動画記録にはOpenCVが必要です。インストールされていない場合は：
```bash
pip install opencv-python
```

### LangGraph Studioで実行

```bash
cd /root/LearnVLM/agents/p02_ex_arm_react
langgraph dev
```

ブラウザで `http://localhost:8123` にアクセスし、エージェントを選択して実行できます。

**注意**: `langgraph dev`を使用する場合は、`langgraph.json`が必要です。このファイルは既に作成済みです。

## 実行フロー

1. ユーザーが「アヒルをトレイに運んで」と指示
2. Agentノードが思考：
   - 「まずアヒルの位置を探そう。find_object("duck") を呼び出す。」
3. ToolNodeが find_object("duck") を実行
   - 結果: 「アヒルの位置は [0.6, 0.3, 0.016] です。」
4. Agentノードが再度思考：
   - 「アヒルの位置がわかった。次はアヒルの10cm上空に移動しよう。move_arm(0.6, 0.3, 0.116) を呼び出す。」
5. ToolNodeが move_arm を実行
6. このループが続き、最終的に：
   - アヒルを掴む（grasp_object）
   - トレイの位置を探す（find_object("tray")）
   - トレイの上に移動（move_arm）
   - アヒルを離す（release_object）
   - 「目標達成！」と発言して終了

## 注意事項

- PyBullet環境はデフォルトでDIRECTモード（GUIなし）で実行されます
- VLMモデル（moondream2）は初回実行時に自動的にロードされます（CUDAが必要）
- エージェントは最大反復回数に制限がないため、適切な終了条件を設定してください
