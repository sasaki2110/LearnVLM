"""
ReActエージェントのテスト
"""
import sys
from pathlib import Path

# このテストファイルの親ディレクトリ（p02_ex_arm_react）をパスに追加
# これにより、pip install -e . なしでも動作する
test_dir = Path(__file__).parent
agent_dir = test_dir.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from langchain.messages import HumanMessage
from my_agent.agent import graph
from my_agent.utils.state import State
import os

def test_invoke():
    """エージェントを呼び出してテスト"""
    print("🧪 [TEST] ReActエージェントのテストを開始します")
    
    # GUIモードの設定（環境変数で制御）
    use_gui = os.getenv("USE_GUI", "false").lower() == "true"
    if use_gui:
        print("🖥️  [TEST] GUIモードが有効です")
        print("   環境変数 USE_GUI=true が設定されています")
        print("   PyBulletのGUIウィンドウが表示されます")
    
    # 動画記録の設定（環境変数で制御）
    # 注意: 動画記録はツール側で自動的に有効化されます
    record_video = os.getenv("RECORD_VIDEO", "false").lower() == "true"
    video_filename = os.getenv("VIDEO_FILENAME", "react_agent_simulation.mp4")
    
    if record_video:
        print(f"🎬 [TEST] 動画記録が有効です: {video_filename}")
        print("   環境変数 RECORD_VIDEO=true が設定されています")
        print("   ツール実行時に自動的に動画記録が開始されます")
    
    # 初期状態
    initial_state: State = {
        "messages": [HumanMessage(content="アヒルをトレイに運んで")],
        "current_arm_position": None,
        "grasped_object": None,
        "last_tool_result": None,
        "tool_history": []
    }
    
    # エージェントを実行（再帰制限を増やす）
    config = {"recursion_limit": 50}  # デフォルトの25から50に増やす
    result = graph.invoke(initial_state, config=config)
    
    print("\n📊 [TEST] 実行結果:")
    print(f"メッセージ数: {len(result.get('messages', []))}")
    print(f"最終アーム位置: {result.get('current_arm_position')}")
    print(f"掴んでいる物体: {result.get('grasped_object')}")
    print(f"ツール履歴: {result.get('tool_history', [])}")
    
    print("\n💬 [TEST] メッセージ履歴:")
    for i, msg in enumerate(result.get('messages', [])):
        print(f"\n--- メッセージ {i+1} ---")
        if hasattr(msg, 'content'):
            print(f"内容: {msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"ツール呼び出し: {msg.tool_calls}")
        print(f"タイプ: {type(msg).__name__}")
    
    print("\n✅ [TEST] テスト完了")

if __name__ == "__main__":
    test_invoke()
