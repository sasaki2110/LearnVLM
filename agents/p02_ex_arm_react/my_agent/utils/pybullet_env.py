"""
PyBullet環境管理モジュール（ReActエージェント用）

KUKA IIWAアーム、アヒル、トレイを配置
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
from typing import Dict, Tuple, Optional, List
from my_agent.utils.logging_config import get_logger

logger = get_logger('pybullet_env')


class PyBulletEnvironment:
    """PyBulletシミュレーション環境の管理クラス（ReAct用）"""
    
    def __init__(self, use_gui: bool = False, record_video: bool = False, video_filename: str = "simulation.mp4"):
        """環境を初期化"""
        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.object_ids: Dict[str, int] = {}
        self.plane_id: Optional[int] = None
        self.end_effector_index = 6  # Kuka iiwaのエンドエフェクタリンクインデックス
        self.use_gui = use_gui
        self.constraints: Dict[str, int] = {}  # 物体名 -> constraint ID
        self.record_video = record_video
        self.video_filename = video_filename
        self.video_writer = None
        self.frames = []  # フレームを一時保存（後で動画に変換）
        
    def create_environment(self) -> int:
        """新しいPyBullet環境を作成"""
        logger.info("🚀 [PYBULLET_ENV] 新しいPyBullet環境を作成します")
        
        try:
            if self.use_gui:
                self.client_id = p.connect(p.GUI)
                logger.info(f"🔗 [PYBULLET_ENV] GUIモードで接続しました (Client ID: {self.client_id})")
            else:
                self.client_id = p.connect(p.DIRECT)
                logger.info(f"🔗 [PYBULLET_ENV] DIRECTモードで接続しました (Client ID: {self.client_id})")
                
            if self.client_id < 0:
                error_msg = "物理サーバーへの接続に失敗しました。"
                logger.error(f"❌ [PYBULLET_ENV] {error_msg}")
                raise RuntimeError(error_msg)
            
            logger.info(f"✅ [PYBULLET_ENV] PyBullet接続成功 (Client ID: {self.client_id}, GUI: {self.use_gui})")
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # 床をロード
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
            logger.debug(f"📦 [PYBULLET_ENV] 床をロードしました (ID: {self.plane_id})")
            
            # KUKA IIWAアームを[0, 0, 0]に固定
            self.robot_id = p.loadURDF(
                "kuka_iiwa/model.urdf",
                [0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.client_id
            )
            logger.debug(f"🤖 [PYBULLET_ENV] KUKA IIWAロボットをロードしました (ID: {self.robot_id})")
            
            # アヒルを左側に固定配置（カメラ視野内に収まるように調整）
            duck_pos = [0.5, -0.2, 0.0]
            duck_id = self.load_object("duck", "duck_vhacd.urdf", duck_pos)
            logger.info(f"🦆 [PYBULLET_ENV] アヒルを左側に配置: {duck_pos}")
            
            # トレイを右側に固定配置（カメラ視野内に収まるように調整）
            tray_pos = [0.6, 0.2, 0.0]
            tray_id = self.load_object("tray", "tray/tray.urdf", tray_pos)
            logger.info(f"📦 [PYBULLET_ENV] トレイを右側に配置: {tray_pos}")
            
            # シミュレーションを500ステップ実行して安定化
            for _ in range(500):
                p.stepSimulation(physicsClientId=self.client_id)
            
            logger.info("✅ [PYBULLET_ENV] 環境の初期化が完了しました")
            return self.client_id
            
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] 環境作成中にエラーが発生しました: {e}", exc_info=True)
            self.cleanup()
            raise
    
    def load_object(self, object_name: str, urdf_path: str, position: Tuple[float, float, float] = (0.0, 0.0, 0.01)) -> int:
        """物体を環境にロード"""
        if self.client_id is None:
            raise RuntimeError("環境が作成されていません。先にcreate_environment()を呼び出してください。")
        
        logger.info(f"📦 [PYBULLET_ENV] 物体をロード: {object_name} at {position}")
        
        try:
            object_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                physicsClientId=self.client_id
            )
            self.object_ids[object_name] = object_id
            logger.debug(f"✅ [PYBULLET_ENV] 物体 '{object_name}' をロードしました (ID: {object_id})")
            return object_id
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] 物体ロード中にエラーが発生しました: {e}", exc_info=True)
            raise
    
    def get_object_position(self, object_name: str) -> Optional[Tuple[float, float, float]]:
        """物体の位置を取得"""
        if self.client_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        if object_name not in self.object_ids:
            logger.warning(f"⚠️ [PYBULLET_ENV] 物体 '{object_name}' が見つかりません")
            return None
        
        object_id = self.object_ids[object_name]
        pos, _ = p.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)
        return pos
    
    def move_arm_to(self, target_position: Tuple[float, float, float], steps: int = 200) -> bool:
        """アームを指定位置に移動（逆運動学を使用）"""
        if self.client_id is None or self.robot_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        logger.info(f"🤖 [PYBULLET_ENV] アームを移動: {target_position}")
        
        try:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                target_position,
                physicsClientId=self.client_id
            )
            
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            for j in range(min(len(joint_poses), num_joints)):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[j],
                    physicsClientId=self.client_id
                )
            
            for _ in range(steps):
                p.stepSimulation(physicsClientId=self.client_id)
            
            link_state = p.getLinkState(
                self.robot_id,
                self.end_effector_index,
                physicsClientId=self.client_id
            )
            actual_pos = link_state[4]
            
            logger.info(f"✅ [PYBULLET_ENV] アーム移動完了: 目標={target_position}, 実際={actual_pos}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] アーム移動中にエラーが発生しました: {e}", exc_info=True)
            return False
    
    def get_arm_position(self) -> Optional[Tuple[float, float, float]]:
        """アームの現在位置を取得"""
        if self.client_id is None or self.robot_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        try:
            link_state = p.getLinkState(
                self.robot_id,
                self.end_effector_index,
                physicsClientId=self.client_id
            )
            pos = link_state[4]
            return pos
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] アーム位置取得中にエラーが発生しました: {e}", exc_info=True)
            return None
    
    def capture_image(self) -> 'Image':
        """カメラ画像をキャプチャ"""
        if self.client_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        import numpy as np
        from PIL import Image
        
        view_matrix = p.computeViewMatrix([1.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.33, 0.1, 100.0)
        _, _, rgb_img, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix, physicsClientId=self.client_id)
        image = Image.fromarray(np.reshape(rgb_img, (480, 640, 4))[:, :, :3].astype('uint8'))
        
        # 動画記録が有効な場合、フレームを保存
        if self.record_video:
            self._save_frame(rgb_img)
        
        return image
    
    def _capture_frame_for_video(self):
        """動画記録用にフレームをキャプチャ"""
        try:
            view_matrix = p.computeViewMatrix([1.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(60, 1.33, 0.1, 100.0)
            _, _, rgb_img, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix, physicsClientId=self.client_id)
            self._save_frame(rgb_img)
        except Exception as e:
            logger.warning(f"⚠️ [PYBULLET_ENV] フレームキャプチャエラー: {e}")
    
    def _save_frame(self, rgb_img: np.ndarray):
        """フレームを保存（動画記録用）"""
        try:
            # RGB画像を取得（RGBAからRGBに変換）
            frame = np.reshape(rgb_img, (480, 640, 4))[:, :, :3]
            self.frames.append(frame.copy())
        except Exception as e:
            logger.warning(f"⚠️ [PYBULLET_ENV] フレーム保存エラー: {e}")
    
    def save_video(self):
        """保存したフレームを動画ファイルに変換"""
        if not self.record_video or not self.frames:
            return
        
        try:
            import cv2
            
            logger.info(f"🎬 [PYBULLET_ENV] 動画を保存中: {self.video_filename} ({len(self.frames)}フレーム)")
            
            # 動画ライターを初期化
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10  # 10 FPS
            height, width = 480, 640
            
            out = cv2.VideoWriter(self.video_filename, fourcc, fps, (width, height))
            
            # 各フレームを書き込み
            for frame in self.frames:
                # RGBからBGRに変換（OpenCVはBGRを使用）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"✅ [PYBULLET_ENV] 動画を保存しました: {self.video_filename}")
            
            # フレームをクリア
            self.frames = []
            
        except ImportError:
            logger.warning("⚠️ [PYBULLET_ENV] OpenCVがインストールされていません。動画を保存できません。")
            logger.warning("   インストール: pip install opencv-python")
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] 動画保存エラー: {e}", exc_info=True)
    
    def create_constraint(self, object_name: str) -> Optional[int]:
        """
        手先の最も近くにある物体をアームに固定（p.createConstraint）
        
        Args:
            object_name: 固定する物体の名前
            
        Returns:
            Constraint ID、失敗時はNone
        """
        if self.client_id is None or self.robot_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        if object_name not in self.object_ids:
            logger.warning(f"⚠️ [PYBULLET_ENV] 物体 '{object_name}' が見つかりません")
            return None
        
        object_id = self.object_ids[object_name]
        arm_pos = self.get_arm_position()
        obj_pos, _ = p.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)
        
        # エンドエフェクタリンクに固定
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.end_effector_index,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,  # 物体のベースリンク
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client_id
        )
        
        self.constraints[object_name] = constraint_id
        logger.info(f"🔗 [PYBULLET_ENV] 物体 '{object_name}' をアームに固定しました (Constraint ID: {constraint_id})")
        return constraint_id
    
    def remove_constraint(self, object_name: str) -> bool:
        """
        固定を解除（p.removeConstraint）
        
        Args:
            object_name: 固定を解除する物体の名前
            
        Returns:
            成功したかどうか
        """
        if self.client_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        if object_name not in self.constraints:
            logger.warning(f"⚠️ [PYBULLET_ENV] 物体 '{object_name}' の固定が見つかりません")
            return False
        
        constraint_id = self.constraints[object_name]
        p.removeConstraint(constraint_id, physicsClientId=self.client_id)
        del self.constraints[object_name]
        logger.info(f"🔓 [PYBULLET_ENV] 物体 '{object_name}' の固定を解除しました")
        return True
    
    def is_grasped(self, object_name: str) -> bool:
        """物体が掴まれているかどうか"""
        return object_name in self.constraints
    
    def cleanup(self):
        """環境をクリーンアップ"""
        # 動画を保存
        if self.record_video and self.frames:
            self.save_video()
        
        if self.client_id is not None:
            try:
                # 接続状態を確認してから切断
                try:
                    p.getConnectionInfo(physicsClientId=self.client_id)
                    p.disconnect(physicsClientId=self.client_id)
                    logger.info(f"✅ [PYBULLET_ENV] 環境をクリーンアップしました (Client ID: {self.client_id})")
                except:
                    # 既に切断されている場合はスキップ
                    logger.debug(f"⚠️ [PYBULLET_ENV] 環境は既に切断されています (Client ID: {self.client_id})")
            except Exception as e:
                logger.error(f"❌ [PYBULLET_ENV] クリーンアップ中にエラーが発生しました: {e}", exc_info=True)
            finally:
                self.client_id = None
                self.robot_id = None
                self.object_ids = {}
                self.plane_id = None
                self.constraints = {}
                self.frames = []


# グローバルな環境インスタンス
_global_env: Optional[PyBulletEnvironment] = None


def get_environment(use_gui: bool = False, force_new: bool = False, record_video: Optional[bool] = None, video_filename: Optional[str] = None) -> PyBulletEnvironment:
    """
    PyBullet環境を取得（既存の環境を再利用）
    
    Args:
        use_gui: GUIモードを使用するかどうか（指定がない場合は環境変数USE_GUIを参照）
        force_new: 強制的に新しい環境を作成するかどうか
    
    Returns:
        PyBulletEnvironmentインスタンス
    """
    global _global_env
    
    # 環境変数からGUI設定を取得（指定がない場合）
    if use_gui is False:
        import os
        use_gui = os.getenv("USE_GUI", "false").lower() == "true"
    
    # 環境変数から動画記録設定を取得（指定されていない場合）
    if record_video is None:
        import os
        record_video = os.getenv("RECORD_VIDEO", "false").lower() == "true"
    
    if video_filename is None:
        import os
        video_filename = os.getenv("VIDEO_FILENAME", "react_agent_simulation.mp4")
    
    # 動画記録が有効な場合、GUIも有効にする
    if record_video and not use_gui:
        logger.info("🎬 [PYBULLET_ENV] 動画記録が有効なため、GUIモードを有効にします")
        use_gui = True
    
    # 既存の環境があり、再利用可能な場合
    if _global_env is not None and not force_new:
        # 環境が有効か確認
        if _global_env.client_id is not None:
            try:
                # 接続状態を確認
                p.getConnectionInfo(physicsClientId=_global_env.client_id)
                # GUI設定が一致しているか確認
                if _global_env.use_gui == use_gui:
                    logger.debug("♻️ [PYBULLET_ENV] 既存の環境を再利用します")
                    return _global_env
                else:
                    # GUI設定が異なる場合は既存の環境をクリーンアップ
                    logger.info(f"🔄 [PYBULLET_ENV] GUI設定が変更されました (既存: {_global_env.use_gui}, 新規: {use_gui})。新しい環境を作成します")
                    _global_env.cleanup()
                    _global_env = None
            except:
                # 接続が切れている場合は新しい環境を作成
                logger.debug("⚠️ [PYBULLET_ENV] 既存の環境が無効です。新しい環境を作成します")
                _global_env = None
    
    # 新しい環境を作成
    if _global_env is not None:
        _global_env.cleanup()
    
    _global_env = PyBulletEnvironment(use_gui=use_gui, record_video=record_video, video_filename=video_filename)
    _global_env.create_environment()
    
    if record_video:
        logger.info(f"🎬 [PYBULLET_ENV] 動画記録を開始します: {video_filename}")
    
    return _global_env
