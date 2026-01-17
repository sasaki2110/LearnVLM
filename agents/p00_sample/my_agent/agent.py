"""
ストリーミング対応のグラフ定義

このグラフは、トピックを精緻化してからジョークを生成するシンプルなストリーミングエージェントです。
Vercel AI SDKのチャットから呼び出して使用できます。
LangGraph Studio（`langgraph dev`）で使用できます。
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenvがインストールされていない場合はスキップ
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from my_agent.utils.state import State
from my_agent.utils.nodes import extract_topic, refine_topic, generate_joke
from my_agent.utils.logging_config import setup_logging, get_logger, get_log_level

# ロギングをセットアップ
log_level = get_log_level()
setup_logging(log_level=log_level, initialize=True)
logger = get_logger('agent')

logger.info("🚀 [AGENT] エージェントの初期化を開始します")

# OpenAI設定
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
logger.info(f"🤖 [AGENT] 使用モデル: {MODEL_NAME}")

try:
    # モデルの初期化
    logger.debug("🤖 [AGENT] チャットモデルを初期化しています...")
    llm = init_chat_model(
        MODEL_NAME,
        temperature=0
    )
    logger.info("✅ [AGENT] チャットモデルの初期化が完了しました")

    # ノード関数をラップ（llmを閉包で保持）
    def extract_topic_node(state: State):
        """メッセージからトピックを抽出するノード"""
        return extract_topic(state)
    
    
    def refine_topic_node(state: State):
        """トピックを精緻化するノード（llmを閉包で保持）"""
        return refine_topic(state, llm)
    
    
    def generate_joke_node(state: State):
        """ジョークを生成するノード（llmを閉包で保持）"""
        return generate_joke(state, llm)
    
    
    # グラフの構築
    logger.debug("📊 [AGENT] グラフの構築を開始します")
    graph = StateGraph(State)
    
    # ノードの追加
    graph.add_node("extract_topic", extract_topic_node)
    graph.add_node("refine_topic", refine_topic_node)
    graph.add_node("generate_joke", generate_joke_node)
    logger.info("✅ [AGENT] ノードの追加が完了しました (extract_topic, refine_topic, generate_joke)")
    
    # エッジの追加
    graph.add_edge(START, "extract_topic")
    graph.add_edge("extract_topic", "refine_topic")
    graph.add_edge("refine_topic", "generate_joke")
    graph.add_edge("generate_joke", END)
    logger.info("✅ [AGENT] エッジの追加が完了しました")
    
    # コンパイルしてモジュールレベルの変数に代入
    # langgraph.jsonでは "./agents/p00_sample/agent.py:graph" として参照可能
    logger.debug("🔨 [AGENT] グラフをコンパイルしています...")
    graph = graph.compile()
    logger.info("✅ [AGENT] エージェントの初期化が完了しました")
    
except Exception as e:
    logger.error(f"❌ [AGENT] エージェントの初期化中にエラーが発生しました: {e}", exc_info=True)
    raise
