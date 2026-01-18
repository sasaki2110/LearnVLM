"""
エージェントのinvokeテスト
"""
import pytest
from langchain.messages import HumanMessage
from my_agent.agent import graph


def test_invoke():
    """基本的なinvokeテスト"""
    initial_state = {
        "messages": [
            HumanMessage(content="VLMロボットブリッジを実行してください")
        ]
    }
    
    result = graph.invoke(initial_state)
    
    assert "messages" in result
    assert len(result["messages"]) > 0
    print(f"✅ テスト成功: {result['messages'][-1].content}")


if __name__ == "__main__":
    test_invoke()
