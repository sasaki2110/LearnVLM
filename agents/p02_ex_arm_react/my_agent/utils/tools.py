"""
ReActエージェント用ツール定義

find_object, move_arm, grasp_object, release_object
"""
import numpy as np
import torch
import re
import json
import threading
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Dict, Tuple, Optional, List
from my_agent.utils.logging_config import get_logger
from my_agent.utils.pybullet_env import get_environment, PyBulletEnvironment

logger = get_logger('tools')


def _get_use_gui() -> bool:
    """環境変数からGUIモードの設定を取得"""
    return os.getenv("USE_GUI", "false").lower() == "true"

# グローバルなVLMモデル（遅延ロード）
_vlm_model = None
_vlm_tokenizer = None
_vlm_lock = threading.Lock()  # スレッドセーフのためのロック


def _load_vlm():
    """VLMモデルをロード（遅延ロード、スレッドセーフ）"""
    global _vlm_model, _vlm_tokenizer
    with _vlm_lock:
        if _vlm_model is None:
            logger.info("🚀 VLMロード中...")
            model_id = "vikhyatk/moondream2"
            # p01_ex_armと同じ方法でロード（直接.to("cuda")をチェーン）
            if torch.cuda.is_available():
                _vlm_model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                ).to("cuda")
            else:
                _vlm_model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
            _vlm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.info("✅ VLMロード完了")
    return _vlm_model, _vlm_tokenizer


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
    y_center = (vlm_coords[0] + vlm_coords[2]) / 2
    x_center = (vlm_coords[1] + vlm_coords[3]) / 2
    
    bbox_width = vlm_coords[3] - vlm_coords[1]
    bbox_height = vlm_coords[2] - vlm_coords[0]
    bbox_size = max(bbox_width, bbox_height)
    
    distance_factor = kwargs.get('distance_factor', 1.0)
    
    world_x = 0.8 - (y_center * 0.4 * distance_factor)
    world_y = 0.4 - (x_center * 0.5 * distance_factor)
    estimated_z = plane_z + 0.016 + (1.0 - bbox_size) * 0.02
    
    result = [float(world_x), float(world_y), float(estimated_z)]
    return result


def find_object(target_name: str) -> Dict[str, any]:
    """
    物体を検出して3D座標を返す
    
    Args:
        target_name: 検出する物体の名前（例: "duck", "tray"）
    
    Returns:
        辞書: {
            "success": bool,
            "position": [x, y, z] or None,
            "error": str or None
        }
    """
    logger.info(f"🔍 [TOOL] find_object を実行: target_name={target_name}")
    
    try:
        env = get_environment(use_gui=_get_use_gui())
        model, tokenizer = _load_vlm()
        
        # 画像をキャプチャ
        img = env.capture_image()
        enc_image = model.encode_image(img)
        
        # VLMで物体を検出（より明確な質問）
        query = f"Return the bounding box coordinates as a list [ymin, xmin, ymax, xmax] for the {target_name}. Only return the numbers, nothing else."
        coords = model.answer_question(enc_image, query, tokenizer)
        
        logger.debug(f"🔍 [TOOL] VLM応答: {coords}")
        
        try:
            # 座標を解析（evalではなく、より堅牢な方法）
            num_coords = None
            
            # パターン1: JSON配列形式 [0.1, 0.2, 0.3, 0.4]
            json_match = re.search(r'\[[\d\.\s,\-]+\]', coords)
            if json_match:
                try:
                    num_coords = json.loads(json_match.group())
                    if isinstance(num_coords, list) and len(num_coords) >= 4:
                        num_coords = [float(x) for x in num_coords[:4]]
                    else:
                        num_coords = None
                except:
                    pass
            
            # パターン2: 数字を4つ抽出してリストにする
            if num_coords is None:
                numbers = re.findall(r'[\d\.]+', coords)
                if len(numbers) >= 4:
                    try:
                        num_coords = [float(n) for n in numbers[:4]]
                    except:
                        pass
            
            # パターン3: 直接eval（安全チェック付き）
            if num_coords is None:
                safe_coords = coords.strip()
                # 数字、括弧、カンマ、スペース、マイナスのみを含むか
                if re.match(r'^[\d\.\s\[\],\-]+$', safe_coords):
                    try:
                        num_coords = eval(safe_coords)
                        if not isinstance(num_coords, list) or len(num_coords) < 4:
                            num_coords = None
                    except:
                        pass
            
            # VLMが失敗した場合、PyBulletから直接位置を取得（フォールバック）
            if num_coords is None or len(num_coords) < 4:
                logger.warning(f"⚠️ [TOOL] VLM座標解析失敗、PyBulletから直接位置を取得します: {coords}")
                obj_pos = env.get_object_position(target_name)
                if obj_pos:
                    # PyBulletから直接位置を取得できた場合
                    logger.info(f"✅ [TOOL] PyBulletから位置を取得: {target_name} = {obj_pos}")
                    return {
                        "success": True,
                        "position": list(obj_pos),
                        "error": None
                    }
                else:
                    raise ValueError(f"座標を抽出できませんでした: {coords}")
            
            bbox_size = max(num_coords[3] - num_coords[1], num_coords[2] - num_coords[0])
            
            # 現在のアーム位置を取得（距離ファクター計算用）
            current_arm_pos = env.get_arm_position()
            if current_arm_pos is not None:
                arm_to_target = np.linalg.norm(np.array(current_arm_pos[:2]) - np.array([0.5, 0.0]))
                distance_factor = max(0.5, min(1.5, arm_to_target / 0.5))
            else:
                distance_factor = 1.0
            
            # 2D座標を3D空間座標に変換
            target_3d = map_vlm_to_world(
                num_coords,
                camera_eye=[1.0, 0.0, 1.0],
                camera_target=[0.5, 0.0, 0.0],
                fov=60,
                image_size=(640, 480),
                plane_z=0.0,
                distance_factor=distance_factor
            )
            
            logger.info(f"✅ [TOOL] find_object 成功: {target_name} の位置 = {target_3d}")
            return {
                "success": True,
                "position": target_3d,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ [TOOL] find_object 座標解析エラー: {coords}, エラー: {e}")
            
            # 最後のフォールバック：PyBulletから直接位置を取得
            try:
                obj_pos = env.get_object_position(target_name)
                if obj_pos:
                    logger.info(f"✅ [TOOL] フォールバック: PyBulletから位置を取得: {target_name} = {obj_pos}")
                    return {
                        "success": True,
                        "position": list(obj_pos),
                        "error": None
                    }
            except:
                pass
            
            return {
                "success": False,
                "position": None,
                "error": f"座標解析に失敗しました: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"❌ [TOOL] find_object 実行中にエラーが発生しました: {e}", exc_info=True)
        return {
            "success": False,
            "position": None,
            "error": str(e)
        }


def move_arm(x: float, y: float, z: float) -> Dict[str, any]:
    """
    アームを指定位置に移動
    
    Args:
        x: X座標
        y: Y座標
        z: Z座標
    
    Returns:
        辞書: {
            "success": bool,
            "current_position": [x, y, z] or None,
            "error": str or None
        }
    """
    logger.info(f"🤖 [TOOL] move_arm を実行: [{x}, {y}, {z}]")
    
    try:
        env = get_environment(use_gui=_get_use_gui())
        target_position = (float(x), float(y), float(z))
        
        success = env.move_arm_to(target_position)
        
        if success:
            current_pos = env.get_arm_position()
            logger.info(f"✅ [TOOL] move_arm 成功: 現在地 = {current_pos}")
            return {
                "success": True,
                "current_position": list(current_pos) if current_pos else None,
                "error": None
            }
        else:
            return {
                "success": False,
                "current_position": None,
                "error": "アームの移動に失敗しました"
            }
            
    except Exception as e:
        logger.error(f"❌ [TOOL] move_arm 実行中にエラーが発生しました: {e}", exc_info=True)
        return {
            "success": False,
            "current_position": None,
            "error": str(e)
        }


def grasp_object() -> Dict[str, any]:
    """
    手先の最も近くにある物体をアームに固定
    
    Returns:
        辞書: {
            "success": bool,
            "grasped_object": str or None,
            "error": str or None
        }
    """
    logger.info("🤏 [TOOL] grasp_object を実行")
    
    try:
        env = get_environment(use_gui=_get_use_gui())
        arm_pos = env.get_arm_position()
        
        if arm_pos is None:
            return {
                "success": False,
                "grasped_object": None,
                "error": "アーム位置が取得できませんでした"
            }
        
        # 手先に最も近い物体を探す
        min_distance = float('inf')
        closest_object = None
        
        for obj_name in env.object_ids.keys():
            obj_pos = env.get_object_position(obj_name)
            if obj_pos:
                distance = np.linalg.norm(np.array(arm_pos) - np.array(obj_pos))
                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj_name
        
        if closest_object is None:
            return {
                "success": False,
                "grasped_object": None,
                "error": "近くに物体が見つかりませんでした"
            }
        
        # 物体を固定
        constraint_id = env.create_constraint(closest_object)
        
        if constraint_id is not None:
            logger.info(f"✅ [TOOL] grasp_object 成功: {closest_object} を掴みました")
            return {
                "success": True,
                "grasped_object": closest_object,
                "error": None
            }
        else:
            return {
                "success": False,
                "grasped_object": None,
                "error": f"{closest_object} の固定に失敗しました"
            }
            
    except Exception as e:
        logger.error(f"❌ [TOOL] grasp_object 実行中にエラーが発生しました: {e}", exc_info=True)
        return {
            "success": False,
            "grasped_object": None,
            "error": str(e)
        }


def release_object() -> Dict[str, any]:
    """
    固定を解除
    
    Returns:
        辞書: {
            "success": bool,
            "released_object": str or None,
            "error": str or None
        }
    """
    logger.info("🔓 [TOOL] release_object を実行")
    
    try:
        env = get_environment(use_gui=_get_use_gui())
        
        # 固定されている物体を探す
        grasped_objects = list(env.constraints.keys())
        
        if not grasped_objects:
            return {
                "success": False,
                "released_object": None,
                "error": "固定されている物体がありません"
            }
        
        # 最初に見つかった物体の固定を解除
        released_object = grasped_objects[0]
        success = env.remove_constraint(released_object)
        
        if success:
            logger.info(f"✅ [TOOL] release_object 成功: {released_object} を離しました")
            return {
                "success": True,
                "released_object": released_object,
                "error": None
            }
        else:
            return {
                "success": False,
                "released_object": None,
                "error": f"{released_object} の固定解除に失敗しました"
            }
            
    except Exception as e:
        logger.error(f"❌ [TOOL] release_object 実行中にエラーが発生しました: {e}", exc_info=True)
        return {
            "success": False,
            "released_object": None,
            "error": str(e)
        }
