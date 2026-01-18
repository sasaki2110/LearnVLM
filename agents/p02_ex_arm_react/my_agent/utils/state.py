"""
ReActエージェントの状態定義
"""
from typing import TypedDict, Annotated, Optional, List, Dict
from langchain.messages import AnyMessage
import operator


class State(TypedDict):
    """ReActエージェントの状態"""
    messages: Annotated[list[AnyMessage], operator.add]  # メッセージ履歴
    current_arm_position: Optional[List[float]]  # 現在のアーム位置 [x, y, z]
    grasped_object: Optional[str]  # 掴んでいる物体の名前（None = 何も掴んでいない）
    last_tool_result: Optional[Dict]  # 前回のツール実行結果
    tool_history: Annotated[List[str], operator.add]  # ツール実行履歴
