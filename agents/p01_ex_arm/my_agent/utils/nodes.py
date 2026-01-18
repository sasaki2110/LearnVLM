"""
ノード関数の実装
"""
from langchain.messages import HumanMessage, AIMessage
from my_agent.utils.state import State
from my_agent.utils.logging_config import get_logger
from my_agent.utils.tools import vlm_robot_bridge

logger = get_logger('nodes')


def vlm_robot_bridge_node(state: State) -> State:
    """
    VLMロボットブリッジツールを実行するノード
    
    Args:
        state: 現在の状態（messagesのみ）
    
    Returns:
        更新された状態
    """
    logger.info("🚀 [NODE] VLMロボットブリッジノードを実行します")
    
    # 最後のユーザーメッセージを取得
    messages = state.get("messages", [])
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    # ツールを実行（GUIモードはデフォルトでFalse）
    use_gui = False
    if user_message and "gui" in user_message.lower():
        use_gui = True
    
    try:
        result = vlm_robot_bridge(use_gui=use_gui)
        
        if result.get("success"):
            response_text = f"✅ VLMロボットブリッジが正常に完了しました。\n"
            response_text += f"🤖 最終的なアームの位置: {result.get('final_arm_position')}\n"
            if result.get('duck_position'):
                response_text += f"🦆 最終的なアヒル位置: {result.get('duck_position')}\n"
            response_text += f"🎯 目標位置: {result.get('target_position')}\n"
            response_text += f"📊 移動回数: {len(result.get('arm_movement_positions', []))}回"
        else:
            response_text = f"❌ VLMロボットブリッジが失敗しました: {result.get('error', '不明なエラー')}"
        
        # 結果をメッセージに追加
        new_messages = [
            AIMessage(content=response_text)
        ]
        
        logger.info("✅ [NODE] VLMロボットブリッジノードの実行が完了しました")
        
        # 状態を更新（位置情報を含む）
        updated_state = {
            "messages": new_messages
        }
        
        # ツールの戻り値から位置情報を取得して状態に追加
        if result.get("success"):
            updated_state["target_position"] = result.get("target_position")
            updated_state["current_arm_position"] = result.get("current_arm_position")
            updated_state["intermediate_target"] = result.get("intermediate_target")
            # arm_movement_positionsはoperator.addで結合されるため、リストとして返す
            # 既存の値がある場合は結合される
            updated_state["arm_movement_positions"] = result.get("arm_movement_positions", [])
            updated_state["duck_position"] = result.get("duck_position")
            
            logger.info(f"📊 [NODE] 状態を更新しました:")
            logger.info(f"  - 目標位置: {updated_state.get('target_position')}")
            logger.info(f"  - 現在のアーム位置: {updated_state.get('current_arm_position')}")
            logger.info(f"  - 中間目標: {updated_state.get('intermediate_target')}")
            logger.info(f"  - アーム移動位置数: {len(updated_state.get('arm_movement_positions', []))}")
            logger.info(f"  - アヒル位置: {updated_state.get('duck_position')}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"❌ [NODE] VLMロボットブリッジノード実行中にエラーが発生しました: {e}", exc_info=True)
        error_message = AIMessage(content=f"❌ エラーが発生しました: {str(e)}")
        return {
            "messages": [error_message]
        }
