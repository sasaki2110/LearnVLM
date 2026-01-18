"""
状態定義（messages + 位置情報）
"""
from typing import TypedDict, Annotated, Optional, List
from langchain.messages import AnyMessage
import operator


class State(TypedDict):
    """エージェントの状態"""
    messages: Annotated[list[AnyMessage], operator.add]  # メッセージ履歴
    target_position: Optional[List[float]]  # 目標位置 [x, y, z]
    current_arm_position: Optional[List[float]]  # 現在のアーム位置 [x, y, z]
    intermediate_target: Optional[List[float]]  # 中間目標位置 [x, y, z]
    arm_movement_positions: Annotated[List[List[float]], operator.add]  # アームの移動位置履歴 [[x, y, z], ...]
    duck_position: Optional[List[float]]  # アヒルの位置 [x, y, z]
