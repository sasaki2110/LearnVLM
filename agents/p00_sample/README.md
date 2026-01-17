# p00_sample - ストリーミング対応のLangGraphエージェント（ひな型）

このプロジェクトは、Vercel AI SDKのチャットから呼び出せるストリーミング対応のLangGraphエージェントのひな型です。
LangGraph Studio（`langgraph dev`）で使用できます。

## 構造

```
p00_sample/
├── utils/             # グラフ用のユーティリティ
│   ├── __init__.py
│   ├── state.py       # グラフの状態定義
│   ├── nodes.py       # グラフ用のノード関数（ログ付き）
│   └── logging_config.py  # ロギング設定
├── tests/             # テストファイル
│   ├── __init__.py
│   └── test_invoke.py    # invokeのテスト
├── __init__.py
├── agent.py          # グラフを構築するコード（ログ付き）
└── README.md         # このファイル
```

## 機能

このエージェントは、以下の処理を行います：

1. **トピック抽出**: メッセージからトピックを抽出します
2. **トピックの精緻化**: 抽出されたトピックを、より面白く魅力的なトピックに精緻化します
3. **ジョーク生成**: 精緻化されたトピックについて、面白いジョークを生成します

## セットアップ

### 1. 依存関係のインストール

プロジェクトルート（`/root/LearnVLM`）で以下のコマンドを実行：

```bash
source .venv/bin/activate
pip install -r requirements_langgraph.txt
```

### 2. パッケージのインストール（重要！）

各エージェントディレクトリで、パッケージをインストールします：

```bash
cd /root/LearnVLM/agents/p00_sample
pip install -e .
```

これにより、`my_agent`パッケージがインストールされ、相対インポートが正しく動作します。

**注意**: 新しいエージェント（`p01_aaa`、`p02_bbb`など）を追加する場合は、各エージェントディレクトリで`pip install -e .`を実行する必要があります。

### 3. 環境変数の設定

`.env` ファイルに以下を設定（プロジェクトルート `/root/LearnVLM/.env`）：

```
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key_here
```

### 4. LangGraph Studioで実行

```bash
cd /root/LearnVLM/agents/p00_sample
langgraph dev
```

サーバーが起動すると、以下のURLでアクセスできます：
- **APIエンドポイント**: `http://127.0.0.1:2024`
- **Studio UI**: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

## Vercel AI SDKからの呼び出し

Vercel AI SDKのチャットから呼び出すには、以下の設定を使用します：

- **API URL**: `http://localhost:2024` (またはデプロイ先のURL)
- **Assistant ID**: `p00_sample`

## テスト

invokeが正常に動作することを確認するテストを実行できます：

```bash
cd /root/LearnVLM
source .venv/bin/activate
python -m pytest agents/p00_sample/tests/test_invoke.py -v
```

または、直接実行：

```bash
cd /root/LearnVLM
source .venv/bin/activate
python agents/p00_sample/tests/test_invoke.py
```

## ロギング機能

### ログファイル

- **通常ログ**: `p00_sample.log` (デフォルト)
- **エラーログ**: `p00_sample_error.log` (ERROR/CRITICALレベルのみ)

### 環境変数

以下の環境変数でロギングをカスタマイズできます：

- `LOG_LEVEL`: ログレベル (DEBUG, INFO, WARNING, ERROR) - デフォルト: INFO
- `LOG_FILE`: ログファイル名 - デフォルト: `p00_sample.log`
- `LOG_DIR`: ログファイルのディレクトリ - デフォルト: `.` (現在のディレクトリ)
- `LOG_USE_PYTHON_ROTATION`: Pythonローテーションを使用するか (true/false) - デフォルト: true
- `ENVIRONMENT`: 環境 (production, development, staging) - デフォルト: development

### ログの内容

自前実装部分（nodes.py, agent.py）に以下のような日本語ログが出力されます：

- **エージェント初期化**: モデルの初期化、グラフの構築
- **トピック抽出**: メッセージからトピックを抽出する処理
- **トピック精緻化**: LLM呼び出し開始/完了、精緻化されたトピック
- **ジョーク生成**: LLM呼び出し開始/完了、生成されたジョーク
- **エラー**: エラー発生時の詳細情報

### ログの確認方法

1. **コンソール出力**: 実行時にコンソールにログが表示されます
2. **ログファイル**: `p00_sample.log` ファイルを確認
3. **エラーログ**: `p00_sample_error.log` ファイルでエラーのみを確認

## 参考

この実装は `/root/LearnLangGraph/archives/p31_streaming` を参考にしています。
