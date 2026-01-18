"""
ReAct型フィジカルAIエージェント

LangGraphを使用したReActエージェント
思考（Thought）→ 行動（Action）→ 観察（Observation）のループ
"""
import os
import sys
from pathlib import Path

# このファイルの親ディレクトリ（p02_ex_arm_react）をパスに追加
# これにより、pip install -e . なしでも動作する
_agent_dir = Path(__file__).parent.parent
if str(_agent_dir) not in sys.path:
    sys.path.insert(0, str(_agent_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from my_agent.utils.state import State
from my_agent.utils.nodes import agent_node, tool_node, should_continue
from my_agent.utils.logging_config import setup_logging, get_logger, get_log_level

# ロギングをセットアップ
log_level = get_log_level()
setup_logging(log_level=log_level, initialize=True)
logger = get_logger('agent')

logger.info("🚀 [AGENT] ReAct型フィジカルAIエージェントの初期化を開始します")

# LLM設定
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
    
    # Agentノードをラップ（llmを閉包で保持）
    def agent_node_wrapper(state: State) -> State:
        return agent_node(state, llm)
    
    # グラフの構築
    logger.debug("📊 [AGENT] グラフの構築を開始します")
    graph = StateGraph(State)
    
    # ノードの追加
    graph.add_node("agent", agent_node_wrapper)
    graph.add_node("tools", tool_node)
    logger.info("✅ [AGENT] ノードの追加が完了しました (agent, tools)")
    
    # エントリーポイント
    graph.set_entry_point("agent")
    
    # 条件分岐エッジ：agent -> tools または end
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # ツール実行後はagentに戻る
    graph.add_edge("tools", "agent")
    
    logger.info("✅ [AGENT] エッジの追加が完了しました")
    
    # コンパイル（再帰制限を設定）
    logger.debug("🔨 [AGENT] グラフをコンパイルしています...")
    graph = graph.compile()
    
    # デフォルトの再帰制限を設定（ReActエージェントは多くのステップが必要な場合がある）
    # 呼び出し時にconfigで上書き可能
    logger.info("✅ [AGENT] ReAct型フィジカルAIエージェントの初期化が完了しました")
    logger.info("💡 [AGENT] 再帰制限: デフォルト25、推奨50以上（config={'recursion_limit': 50}で設定）")
    
except Exception as e:
    logger.error(f"❌ [AGENT] エージェントの初期化中にエラーが発生しました: {e}", exc_info=True)
    raise
