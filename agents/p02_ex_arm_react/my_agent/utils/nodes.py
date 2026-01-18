"""
ReActエージェントのノード実装

Agentノード（思考）とToolNode（ツール実行）
"""
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from my_agent.utils.state import State
from my_agent.utils.logging_config import get_logger
from my_agent.utils.tools import find_object, move_arm, grasp_object, release_object
from my_agent.utils.pybullet_env import get_environment

logger = get_logger('nodes')


# ツールをLangChainのTool形式に変換
@tool
def find_object_tool(target_name: str) -> str:
    """
    物体を検出して3D座標を返す
    
    Args:
        target_name: 検出する物体の名前（例: "duck", "tray"）
    
    Returns:
        結果の文字列（JSON形式）
    """
    result = find_object(target_name)
    if result["success"]:
        pos = result["position"]
        return f"物体 '{target_name}' の位置は [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] です。"
    else:
        return f"エラー: {result.get('error', '不明なエラー')}"


@tool
def move_arm_tool(x: float, y: float, z: float) -> str:
    """
    アームを指定位置に移動
    
    Args:
        x: X座標
        y: Y座標
        z: Z座標
    
    Returns:
        結果の文字列
    """
    result = move_arm(x, y, z)
    if result["success"]:
        pos = result["current_position"]
        return f"移動完了。現在地は [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] です。"
    else:
        return f"エラー: {result.get('error', '不明なエラー')}"


@tool
def grasp_object_tool() -> str:
    """
    手先の最も近くにある物体をアームに固定
    
    Returns:
        結果の文字列
    """
    result = grasp_object()
    if result["success"]:
        obj = result["grasped_object"]
        return f"物体 '{obj}' を掴みました。"
    else:
        return f"エラー: {result.get('error', '不明なエラー')}"


@tool
def release_object_tool() -> str:
    """
    固定を解除
    
    Returns:
        結果の文字列
    """
    result = release_object()
    if result["success"]:
        obj = result["released_object"]
        return f"物体 '{obj}' を離しました。"
    else:
        return f"エラー: {result.get('error', '不明なエラー')}"


# ツールリスト
tools = [find_object_tool, move_arm_tool, grasp_object_tool, release_object_tool]

# ToolNodeを作成
_base_tool_node = ToolNode(tools)


def tool_node(state: State) -> State:
    """
    ToolNodeをラップして、ツール実行後に状態を更新
    
    Args:
        state: 現在の状態
    
    Returns:
        更新された状態
    """
    logger.info("🔧 [NODE] ToolNode（ツール実行）を実行します")
    
    # ベースのToolNodeを実行
    updated_state = _base_tool_node.invoke(state)
    
    # ツール実行結果を状態に反映
    messages = updated_state.get("messages", [])
    if messages:
        # 最後のメッセージがToolMessageか確認
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage):
            tool_name = last_msg.name
            tool_result = last_msg.content
            
            # last_tool_resultを更新
            updated_state["last_tool_result"] = {
                "tool_name": tool_name,
                "result": tool_result
            }
            
            # tool_historyに追加
            updated_state["tool_history"] = [tool_name]
            
            # grasp_object/release_objectの場合はgrasped_objectを更新
            try:
                env = get_environment(use_gui=False)
                if tool_name == "grasp_object_tool" and "掴みました" in tool_result:
                    # 掴んだ物体を抽出（例: "物体 'duck' を掴みました。"）
                    import re
                    match = re.search(r"物体 '(\w+)'", tool_result)
                    if match:
                        updated_state["grasped_object"] = match.group(1)
                elif tool_name == "release_object_tool" and "離しました" in tool_result:
                    updated_state["grasped_object"] = None
                
                # アーム位置を更新
                arm_pos = env.get_arm_position()
                if arm_pos:
                    updated_state["current_arm_position"] = list(arm_pos)
            except:
                pass
            
            logger.info(f"✅ [NODE] ツール '{tool_name}' の実行が完了しました")
    
    return updated_state


def agent_node(state: State, llm) -> State:
    """
    Agentノード（思考）：LLMが状況を見て、思考と行動を決定
    
    Args:
        state: 現在の状態
        llm: LLMモデル
    
    Returns:
        更新された状態
    """
    logger.info("🧠 [NODE] Agentノード（思考）を実行します")
    
    messages = state.get("messages", [])
    current_arm_pos = state.get("current_arm_position")
    grasped_obj = state.get("grasped_object")
    last_tool_result = state.get("last_tool_result")
    
    # システムプロンプトを構築
    system_prompt = """あなたはロボットアームを制御するReAct型AIエージェントです。

最終目標: アヒルをトレイに運んでください。

使えるツール:
- find_object(target_name: str): 物体を検出して3D座標を返す
- move_arm(x, y, z): アームを指定位置に移動
- grasp_object(): 手先の最も近くにある物体をアームに固定
- release_object(): 固定を解除

現在の状況:
"""
    
    if current_arm_pos:
        system_prompt += f"- 現在のアーム位置は [{current_arm_pos[0]:.3f}, {current_arm_pos[1]:.3f}, {current_arm_pos[2]:.3f}] です\n"
    else:
        system_prompt += "- 現在のアーム位置は不明です\n"
    
    if grasped_obj:
        system_prompt += f"- 現在 '{grasped_obj}' を掴んでいます\n"
    else:
        system_prompt += "- 何も掴んでいません\n"
    
    if last_tool_result:
        system_prompt += f"- 前回のツール '{last_tool_result.get('tool_name')}' の結果: {last_tool_result.get('result', '')}\n"
    
    system_prompt += "\n思考（Thought）と行動（Action）を出力してください。目標達成したら「目標達成！」と発言してください。"
    
    # システムメッセージを追加
    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    # LLMを呼び出し（ツール付き）
    response = llm.bind_tools(tools).invoke(full_messages)
    
    logger.info(f"🤖 [NODE] LLM応答: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"🔧 [NODE] ツール呼び出し: {[tc['name'] for tc in response.tool_calls]}")
    
    # 状態を更新
    updated_state = {
        "messages": [response]
    }
    
    # アーム位置を更新（環境から取得）
    try:
        env = get_environment(use_gui=False)
        arm_pos = env.get_arm_position()
        if arm_pos:
            updated_state["current_arm_position"] = list(arm_pos)
    except:
        pass
    
    return updated_state


def should_continue(state: State) -> str:
    """
    条件分岐：ツールを実行するか、終了するか
    
    Args:
        state: 現在の状態
    
    Returns:
        "continue" または "end"
    """
    messages = state.get("messages", [])
    if not messages:
        logger.debug("🔍 [NODE] should_continue: メッセージがありません")
        return "end"
    
    latest = messages[-1]
    
    # AIMessageにtool_callsがあるか確認
    if hasattr(latest, 'tool_calls') and latest.tool_calls:
        logger.debug(f"🔍 [NODE] should_continue: tool_callsがあります: {latest.tool_calls}")
        return "continue"
    
    # 目標達成のキーワードをチェック
    content = latest.content if hasattr(latest, 'content') else str(latest)
    logger.debug(f"🔍 [NODE] should_continue: 最新メッセージの内容: {content}")
    
    # 目標達成のキーワードをチェック（より広範囲に）
    goal_keywords = ["目標達成", "完了", "終了", "成功", "達成", "task completed", "goal achieved"]
    content_lower = content.lower() if isinstance(content, str) else str(content).lower()
    
    for keyword in goal_keywords:
        if keyword in content or keyword in content_lower:
            logger.info(f"✅ [NODE] should_continue: 目標達成を検出しました（キーワード: {keyword}）")
            return "end"
    
    # デフォルトは継続（LLMが再度思考する）
    logger.debug("🔍 [NODE] should_continue: 継続します")
    return "continue"
