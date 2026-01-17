"""
状態定義
"""
from typing import TypedDict, Optional, Annotated
from langchain.messages import AnyMessage
import operator


class State(TypedDict):
    """エージェントの状態"""
    messages: Annotated[list[AnyMessage], operator.add]  # Vercel AI SDKからの入力と出力
    topic: Optional[str]  # 抽出されたトピック
    joke: Optional[str]  # 生成されたジョーク
