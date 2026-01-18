"""
VLMロボットブリッジエージェント

このグラフは、VLMを使用してロボットアームを制御するシンプルなエージェントです。
start -> vlm_robot_bridge_node -> end
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenvがインストールされていない場合はスキップ
from langgraph.graph import StateGraph, START, END
from my_agent.utils.state import State
from my_agent.utils.nodes import vlm_robot_bridge_node
from my_agent.utils.logging_config import setup_logging, get_logger, get_log_level

# ロギングをセットアップ
log_level = get_log_level()
setup_logging(log_level=log_level, initialize=True)
logger = get_logger('agent')

logger.info("🚀 [AGENT] VLMロボットブリッジエージェントの初期化を開始します")

try:
    # グラフの構築
    logger.debug("📊 [AGENT] グラフの構築を開始します")
    graph = StateGraph(State)
    
    # ノードの追加
    graph.add_node("vlm_robot_bridge", vlm_robot_bridge_node)
    logger.info("✅ [AGENT] ノードの追加が完了しました (vlm_robot_bridge)")
    
    # エッジの追加
    graph.add_edge(START, "vlm_robot_bridge")
    graph.add_edge("vlm_robot_bridge", END)
    logger.info("✅ [AGENT] エッジの追加が完了しました (start -> vlm_robot_bridge -> end)")
    
    # コンパイルしてモジュールレベルの変数に代入
    logger.debug("🔨 [AGENT] グラフをコンパイルしています...")
    graph = graph.compile()
    logger.info("✅ [AGENT] VLMロボットブリッジエージェントの初期化が完了しました")
    
except Exception as e:
    logger.error(f"❌ [AGENT] エージェントの初期化中にエラーが発生しました: {e}", exc_info=True)
    raise
