"""
invokeのテスト

このテストは、グラフが正常にinvokeできることを確認します。
ストリーミングではなく、通常のinvokeを使用してテストします。

注意: このテストを実行するには、以下の依存関係がインストールされている必要があります:
- langchain
- langchain-openai
- langgraph
- python-dotenv (オプション)
"""
import sys
from pathlib import Path

# p00_sampleディレクトリをパスに追加
p00_dir = Path(__file__).parent.parent
sys.path.insert(0, str(p00_dir))  # agents/p00_sampleを追加

# テスト対象のグラフをインポート
from my_agent.agent import graph


def test_invoke():
    """グラフが正常にinvokeできることを確認するテスト"""
    from langchain.messages import HumanMessage
    
    # テスト用の入力（Vercel AI SDKのチャット形式を想定）
    initial_state = {
        "messages": [HumanMessage(content="アイスクリーム")],
        "topic": None,
        "joke": None
    }
    
    print("=" * 60)
    print("invokeテスト開始")
    print("=" * 60)
    print(f"\n初期状態:")
    print(f"  messages: {[msg.content for msg in initial_state['messages']]}")
    print(f"  topic: {initial_state['topic']}")
    print(f"  joke: {initial_state['joke']}")
    print("\n" + "-" * 60)
    
    # invokeを実行
    result = graph.invoke(initial_state)
    
    print("\n実行結果:")
    print("-" * 60)
    print(f"  topic: {result.get('topic', 'N/A')}")
    print(f"  joke: {result.get('joke', 'N/A')}")
    print("\n" + "=" * 60)
    
    # 結果の検証
    assert "topic" in result, "結果に'topic'が含まれている必要があります"
    assert "joke" in result, "結果に'joke'が含まれている必要があります"
    assert result["topic"] is not None, "トピックが設定されている必要があります"
    assert len(result["topic"]) > 0, "トピックが空でない必要があります"
    assert result["joke"] is not None, "ジョークが設定されている必要があります"
    assert len(result["joke"]) > 0, "ジョークが生成されている必要があります"
    
    print("✓ すべての検証が成功しました")
    print("=" * 60)
    
    # pytestのテスト関数はNoneを返すべき


if __name__ == "__main__":
    # 直接実行時もNoneを返す（pytestの警告を避けるため）
    test_invoke()
