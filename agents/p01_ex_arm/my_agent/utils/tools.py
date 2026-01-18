"""
VLMロボットブリッジツール

vlm_robot_bridge.pyをベースにしたツール関数
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
from typing import Dict, Tuple, Optional
from my_agent.utils.logging_config import get_logger
from my_agent.utils.pybullet_env import get_environment, PyBulletEnvironment

logger = get_logger('tools')


def map_vlm_to_world(vlm_coords, camera_eye=[1.0, 0.0, 1.0], camera_target=[0.5, 0.0, 0.0], 
                     fov=60, image_size=(640, 480), plane_z=0.0, **kwargs):
    """
    簡易マッピング（2D VLM座標を3Dワールド座標に変換）
    
    Args:
        vlm_coords: [ymin, xmin, ymax, xmax] の正規化座標 (0~1)
        camera_eye: カメラの位置 [x, y, z]
        camera_target: カメラが向く先 [x, y, z]
        fov: 視野角（度）
        image_size: (width, height)
        plane_z: 物体が置かれている平面のZ座標
        **kwargs: 追加パラメータ（distance_factorなど）
    
    Returns:
        ワールド座標 [x, y, z]
    """
    # vlm_coords = [ymin, xmin, ymax, xmax] 
    y_center = (vlm_coords[0] + vlm_coords[2]) / 2
    x_center = (vlm_coords[1] + vlm_coords[3]) / 2
    
    # バウンディングボックスのサイズから奥行きを推定
    bbox_width = vlm_coords[3] - vlm_coords[1]  # x方向のサイズ
    bbox_height = vlm_coords[2] - vlm_coords[0]  # y方向のサイズ
    bbox_size = max(bbox_width, bbox_height)  # 大きい方を使用
    
    # 距離ファクターを受け取る（デフォルトは1.0）
    distance_factor = kwargs.get('distance_factor', 1.0)
    
    # 元のパラメータをベースに、より広い範囲をカバーするように調整
    world_x = 0.8 - (y_center * 0.4 * distance_factor)  # 範囲 [0.4, 0.8] に調整（より広く）
    world_y = 0.4 - (x_center * 0.5 * distance_factor)  # 範囲 [-0.1, 0.4] に調整（より広く）
    
    # Z座標は平面の高さ + アヒルの実際の高さ
    estimated_z = plane_z + 0.016 + (1.0 - bbox_size) * 0.02  # 補正項を小さく
    
    result = [float(world_x), float(world_y), float(estimated_z)]
    return result


def detect_duck_position(env: PyBulletEnvironment, model, tokenizer, show_debug=False, current_arm_pos=None):
    """
    VLMを使ってアヒルの位置を検出
    
    Args:
        env: PyBullet環境
        model: VLMモデル
        tokenizer: VLMトークナイザー
        show_debug: デバッグ情報を表示するか
        current_arm_pos: 現在のアーム位置（距離ファクター計算用）
    
    Returns:
        (target_3d, num_coords, confidence, bbox_size, estimated_distance) または (None, None, 0.0, 0.0, 1.0)
    """
    img = env.capture_image()
    enc_image = model.encode_image(img)
    coords = model.answer_question(enc_image, "Point out the duck with a bounding box.", tokenizer)
    
    try:
        num_coords = eval(coords)
        bbox_size = max(num_coords[3] - num_coords[1], num_coords[2] - num_coords[0])
        
        # バウンディングボックスのサイズから信頼度を計算
        confidence = min(bbox_size * 2.0, 1.0)  # 0~1の範囲に正規化
        
        # アームが近い場合、マッピングのパラメータを調整
        if current_arm_pos is not None:
            arm_to_target = np.linalg.norm(np.array(current_arm_pos[:2]) - np.array([0.5, 0.0]))
            distance_factor = max(0.5, min(1.5, arm_to_target / 0.5))  # 0.5mを基準に
        else:
            distance_factor = 1.0
        
        # マッピングを実行（距離に応じて調整）
        target_3d = map_vlm_to_world(
            num_coords,
            camera_eye=[1.0, 0.0, 1.0],
            camera_target=[0.5, 0.0, 0.0],
            fov=60,
            image_size=(640, 480),
            plane_z=0.0,
            distance_factor=distance_factor
        )
        
        if show_debug:
            y_center = (num_coords[0] + num_coords[2]) / 2
            x_center = (num_coords[1] + num_coords[3]) / 2
            logger.debug(f"🔍 マッピング詳細: y_center={y_center:.3f}, x_center={x_center:.3f}, "
                  f"bbox_size={bbox_size:.3f}, confidence={confidence:.3f}, "
                  f"distance_factor={distance_factor:.3f}, world=[{target_3d[0]:.3f}, {target_3d[1]:.3f}, {target_3d[2]:.3f}]")
        
        # バウンディングボックスのサイズから距離を推定
        estimated_distance = 0.5 / (bbox_size + 0.01)  # 0除算を避けるため+0.01
        estimated_distance = max(0.05, min(1.0, estimated_distance))  # 0.05m~1.0mの範囲に制限
        
        return target_3d, num_coords, confidence, bbox_size, estimated_distance
    except Exception as e:
        logger.error(f"❌ 座標がうまく取れませんでした: {coords}, エラー: {e}")
        return None, None, 0.0, 0.0, 1.0


def vlm_robot_bridge(use_gui: bool = False) -> Dict[str, any]:
    """
    VLMロボットブリッジツール（vlm_robot_bridge.py相当）
    
    VLMを使用してアヒルを検出し、ロボットアームで段階的に接近する
    
    Args:
        use_gui: GUIモードを使用するかどうか（デフォルト: False）
    
    Returns:
        実行結果の辞書
    """
    logger.info("🚀 [TOOL] VLMロボットブリッジを開始します")
    
    try:
        # PyBullet環境を作成
        env = get_environment(use_gui=use_gui)
        
        # アヒルを配置
        duck_target_pos = [0.6, 0.3, 0.0]
        duck_id = env.load_object("duck", "duck_vhacd.urdf", duck_target_pos)
        
        # VLMロード前にアヒルの位置をロギング
        duck_pos = env.get_object_position("duck")
        if duck_pos:
            logger.info(f"📍 VLMロード前のアヒル位置: [{duck_pos[0]:.4f}, {duck_pos[1]:.4f}, {duck_pos[2]:.4f}]")
        else:
            logger.info(f"📍 初期設定位置: [{duck_target_pos[0]:.4f}, {duck_target_pos[1]:.4f}, {duck_target_pos[2]:.4f}]")
        
        logger.info("🚀 VLMロード中...")
        model_id = "vikhyatk/moondream2"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("✅ VLMロード完了")
        
        # ステップ1: VLMで目標位置を決定
        logger.info("📸 ステップ1: VLMで目標位置を決定")
        current_pos = env.get_arm_position()
        result = detect_duck_position(env, model, tokenizer, show_debug=True, current_arm_pos=current_pos)
        if result[0] is None:
            logger.error("❌ 目標位置の検出に失敗しました")
            return {"success": False, "error": "目標位置の検出に失敗しました"}
        
        target_3d, num_coords, confidence, _, _ = result
        logger.info(f"🎯 目標位置: [{target_3d[0]:.4f}, {target_3d[1]:.4f}, {target_3d[2]:.4f}]")
        
        # 目標位置の10cm上を計算
        approach_height = 0.10  # 10cm
        target_above = [target_3d[0], target_3d[1], target_3d[2] + approach_height]
        logger.info(f"📍 接近位置（目標の{approach_height*100:.0f}cm上）: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
        
        # 位置情報を収集するリスト
        arm_movement_positions = []
        intermediate_targets = []
        current_positions = []
        duck_positions = []
        
        # ステップ2: 目標位置の10cm上まで段階的に移動（10段階）
        logger.info(f"📈 ステップ2: 目標位置の{approach_height*100:.0f}cm上まで段階的に移動（10段階）")
        num_steps_approach = 10
        
        for step in range(num_steps_approach):
            logger.info(f"--- 段階 {step + 1}/{num_steps_approach} ---")
            current_pos = env.get_arm_position()
            
            # 各段階でVLMで位置取得をやり直す
            result = detect_duck_position(env, model, tokenizer, show_debug=False, current_arm_pos=current_pos)
            if result[0] is None:
                logger.warning("⚠️ 検出失敗、前回の目標位置を使用")
            else:
                new_target, _, _, _, _ = result
                target_above[0] = new_target[0]
                target_above[1] = new_target[1]
                logger.info(f"🔄 目標位置を更新: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
            
            # 現在位置から目標位置への進捗を計算
            progress = (step + 1) / num_steps_approach
            intermediate_pos = [
                current_pos[i] + (target_above[i] - current_pos[i]) * progress
                for i in range(3)
            ]
            
            logger.info(f"📍 現在位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
            logger.info(f"🎯 中間目標: [{intermediate_pos[0]:.4f}, {intermediate_pos[1]:.4f}, {intermediate_pos[2]:.4f}]")
            
            # 位置情報を収集
            current_positions.append(list(current_pos))
            intermediate_targets.append(list(intermediate_pos))
            
            # 移動
            env.move_arm_to(tuple(intermediate_pos))
            time.sleep(0.3)
            
            # 移動後の位置を記録
            moved_pos = env.get_arm_position()
            if moved_pos:
                arm_movement_positions.append(list(moved_pos))
            
            # アヒルの位置を記録
            duck_pos = env.get_object_position("duck")
            if duck_pos:
                duck_positions.append(list(duck_pos))
        
        # ステップ3: 10cm上に到達したら、真上に移動するまでループ
        logger.info("🎯 ステップ3: 真上に移動するまでループ")
        max_align_iterations = 10
        align_threshold = 0.02  # 2cm以内なら真上とみなす
        
        for align_iter in range(max_align_iterations):
            logger.info(f"--- 位置合わせ {align_iter + 1}/{max_align_iterations} ---")
            current_pos = env.get_arm_position()
            
            # VLMで位置取得
            result = detect_duck_position(env, model, tokenizer, show_debug=False, current_arm_pos=current_pos)
            if result[0] is None:
                logger.error("❌ 検出失敗、終了します")
                break
            
            new_target, _, _, _, _ = result
            
            # X, Y方向の誤差を計算
            xy_error = np.sqrt((current_pos[0] - new_target[0])**2 + (current_pos[1] - new_target[1])**2)
            logger.info(f"📏 X, Y方向の誤差: {xy_error:.4f}m")
            
            if xy_error < align_threshold:
                logger.info(f"✅ 真上に到達しました！（誤差: {xy_error:.4f}m < {align_threshold:.3f}m）")
                break
            
            # X, Yのみを更新（Zは10cm上を維持）
            target_above[0] = new_target[0]
            target_above[1] = new_target[1]
            
            # 少しずつ移動（70%の距離）
            intermediate_pos = [
                current_pos[i] + (target_above[i] - current_pos[i]) * 0.7
                for i in range(3)
            ]
            intermediate_pos[2] = target_above[2]  # Zは常に10cm上
            
            logger.info(f"🎯 目標位置: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
            
            # 位置情報を収集
            current_positions.append(list(current_pos))
            intermediate_targets.append(list(intermediate_pos))
            
            env.move_arm_to(tuple(intermediate_pos))
            time.sleep(0.3)
            
            # 移動後の位置を記録
            moved_pos = env.get_arm_position()
            if moved_pos:
                arm_movement_positions.append(list(moved_pos))
            
            # アヒルの位置を記録
            duck_pos = env.get_object_position("duck")
            if duck_pos:
                duck_positions.append(list(duck_pos))
        
        # ステップ4: 真上に来たら、少しずつ下がる（10段階）
        logger.info("📉 ステップ4: 少しずつ下がる（バウンディングボックスから距離を推定）")
        num_steps_descend = 10
        final_target = target_3d.copy()
        
        current_pos = env.get_arm_position()
        start_z = current_pos[2]
        
        # バウンディングボックスのサイズから適切な停止位置を計算
        result = detect_duck_position(env, model, tokenizer, show_debug=False, current_arm_pos=current_pos)
        if result[0] is not None:
            _, _, _, bbox_size, estimated_distance = result
            safety_margin = max(0.03, min(0.08, estimated_distance * 0.15))  # 3cm~8cmの範囲
            end_z = final_target[2] + safety_margin
            logger.info(f"📏 推定距離: {estimated_distance:.3f}m, バウンディングボックスサイズ: {bbox_size:.3f}")
            logger.info(f"🛡️ 安全マージン: {safety_margin*100:.1f}cm, 停止位置Z: {end_z:.4f}m")
        else:
            safety_margin = 0.05
            end_z = final_target[2] + safety_margin
            logger.warning("⚠️ 検出失敗、デフォルトの安全マージン（5cm）を使用")
        
        for step in range(num_steps_descend):
            logger.info(f"--- 下降段階 {step + 1}/{num_steps_descend} ---")
            current_pos = env.get_arm_position()
            
            # 各段階でVLMで位置取得をやり直す
            result = detect_duck_position(env, model, tokenizer, show_debug=False, current_arm_pos=current_pos)
            if result[0] is not None:
                new_target, _, _, bbox_size, estimated_distance = result
                final_target[0] = new_target[0]
                final_target[1] = new_target[1]
                
                new_safety_margin = max(0.03, min(0.08, estimated_distance * 0.15))
                new_end_z = final_target[2] + new_safety_margin
                
                if new_end_z > end_z:
                    end_z = new_end_z
                    safety_margin = new_safety_margin
                
                logger.info(f"🔄 目標位置を更新: [{final_target[0]:.4f}, {final_target[1]:.4f}, {final_target[2]:.4f}]")
            
            # Z方向の進捗を計算
            progress = (step + 1) / num_steps_descend
            target_z = start_z - (start_z - end_z) * progress
            intermediate_pos = [final_target[0], final_target[1], target_z]
            
            logger.info(f"📍 現在位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
            logger.info(f"🎯 中間目標: [{intermediate_pos[0]:.4f}, {intermediate_pos[1]:.4f}, {intermediate_pos[2]:.4f}]")
            
            if target_z < end_z:
                target_z = end_z
                intermediate_pos[2] = end_z
                logger.warning(f"⚠️ 安全マージンを維持: Z={end_z:.4f}m")
            
            # 位置情報を収集
            current_positions.append(list(current_pos))
            intermediate_targets.append(list(intermediate_pos))
            
            env.move_arm_to(tuple(intermediate_pos))
            time.sleep(0.3)
            
            # 移動後の位置を記録
            moved_pos = env.get_arm_position()
            if moved_pos:
                arm_movement_positions.append(list(moved_pos))
            
            # アヒルの位置を記録
            duck_pos = env.get_object_position("duck")
            if duck_pos:
                duck_positions.append(list(duck_pos))
        
        # 最終結果を取得
        final_pos = env.get_arm_position()
        duck_pos_final = env.get_object_position("duck")
        
        logger.info("📊 最終結果")
        logger.info(f"🤖 最終的なアームの位置: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
        if duck_pos_final:
            logger.info(f"🦆 最終的なアヒル位置: [{duck_pos_final[0]:.4f}, {duck_pos_final[1]:.4f}, {duck_pos_final[2]:.4f}]")
        
        # 環境をクリーンアップ
        env.cleanup()
        
        return {
            "success": True,
            "target_position": target_3d,
            "final_arm_position": final_pos,
            "current_arm_position": final_pos,  # 最終位置が現在位置
            "intermediate_target": intermediate_targets[-1] if intermediate_targets else None,  # 最後の中間目標
            "arm_movement_positions": arm_movement_positions,  # すべての移動位置
            "duck_position": duck_pos_final,
            "duck_positions": duck_positions if duck_positions else [duck_pos_final] if duck_pos_final else [],  # すべてのアヒル位置
            "current_positions": current_positions,  # すべての現在位置
            "intermediate_targets": intermediate_targets  # すべての中間目標
        }
        
    except Exception as e:
        logger.error(f"❌ [TOOL] VLMロボットブリッジ実行中にエラーが発生しました: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
