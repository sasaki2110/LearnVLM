"""
PyBullet環境管理モジュール

PyBulletシミュレーション環境の作成・管理を行う
"""
import pybullet as p
import pybullet_data
from typing import Dict, Tuple, Optional, List
from my_agent.utils.logging_config import get_logger

logger = get_logger('pybullet_env')


class PyBulletEnvironment:
    """PyBulletシミュレーション環境の管理クラス"""
    
    def __init__(self, use_gui: bool = False):
        """環境を初期化（接続は行わない）"""
        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.object_ids: Dict[str, int] = {}  # 物体名 -> 物体ID
        self.plane_id: Optional[int] = None
        self.end_effector_index = 6  # Kuka iiwaのエンドエフェクタリンクインデックス
        self.use_gui = use_gui
        
    def create_environment(self) -> int:
        """
        新しいPyBullet環境を作成
        
        Args:
            use_gui: GUIモードを使用するかどうか（デフォルト: False = DIRECTモード）
        
        Returns:
            クライアントID
        """
        logger.info("🚀 [PYBULLET_ENV] 新しいPyBullet環境を作成します")
        
        try:
            # GUIモードまたはDIRECTモードで接続
            if self.use_gui:
                self.client_id = p.connect(p.GUI)
            else:
                self.client_id = p.connect(p.DIRECT)
            
            if self.client_id < 0:
                error_msg = "物理サーバーへの接続に失敗しました。"
                logger.error(f"❌ [PYBULLET_ENV] {error_msg}")
                raise RuntimeError(error_msg)
            
            logger.info(f"✅ [PYBULLET_ENV] PyBullet接続成功 (Client ID: {self.client_id}, GUI: {self.use_gui})")
            
            # PyBulletに付属している標準データのパスを追加
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # 重力を地球（-9.81）に設定
            p.setGravity(0, 0, -9.81)
            
            # 床をロード
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
            logger.debug(f"📦 [PYBULLET_ENV] 床をロードしました (ID: {self.plane_id})")
            
            # Kuka iiwaロボットをロード（土台を地面に固定）
            self.robot_id = p.loadURDF(
                "kuka_iiwa/model.urdf",
                [0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.client_id
            )
            logger.debug(f"🤖 [PYBULLET_ENV] Kuka iiwaロボットをロードしました (ID: {self.robot_id})")
            
            # シミュレーションを数ステップ実行して安定化
            for _ in range(500):
                p.stepSimulation(physicsClientId=self.client_id)
            
            logger.info("✅ [PYBULLET_ENV] 環境の初期化が完了しました")
            return self.client_id
            
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] 環境作成中にエラーが発生しました: {e}", exc_info=True)
            self.cleanup()
            raise
    
    def load_object(self, object_name: str, urdf_path: str, position: Tuple[float, float, float] = (0.0, 0.0, 0.01)) -> int:
        """
        物体を環境にロード
        
        Args:
            object_name: 物体の名前（識別用）
            urdf_path: URDFファイルのパス（pybullet_dataからの相対パス）
            position: 物体の初期位置 (x, y, z)
            
        Returns:
            物体ID
        """
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
        """
        物体の位置を取得
        
        Args:
            object_name: 物体の名前
            
        Returns:
            物体の位置 (x, y, z)、見つからない場合はNone
        """
        if self.client_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        if object_name not in self.object_ids:
            logger.warning(f"⚠️ [PYBULLET_ENV] 物体 '{object_name}' が見つかりません")
            return None
        
        object_id = self.object_ids[object_name]
        pos, _ = p.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)
        logger.debug(f"📍 [PYBULLET_ENV] 物体 '{object_name}' の位置: {pos}")
        return pos
    
    def move_arm_to(self, target_position: Tuple[float, float, float], steps: int = 200) -> bool:
        """
        アームを指定位置に移動（逆運動学を使用）
        
        Args:
            target_position: 目標位置 (x, y, z)
            steps: シミュレーションステップ数
            
        Returns:
            移動が成功したかどうか
        """
        if self.client_id is None or self.robot_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        logger.info(f"🤖 [PYBULLET_ENV] アームを移動: {target_position}")
        
        try:
            # 逆運動学を計算
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                target_position,
                physicsClientId=self.client_id
            )
            
            # 各関節を目標角度に設定
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            for j in range(min(len(joint_poses), num_joints)):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[j],
                    physicsClientId=self.client_id
                )
            
            # シミュレーションを実行
            for _ in range(steps):
                p.stepSimulation(physicsClientId=self.client_id)
            
            # 実際の位置を確認
            link_state = p.getLinkState(
                self.robot_id,
                self.end_effector_index,
                physicsClientId=self.client_id
            )
            actual_pos = link_state[4]  # 世界座標系での位置
            
            logger.info(f"✅ [PYBULLET_ENV] アーム移動完了: 目標={target_position}, 実際={actual_pos}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] アーム移動中にエラーが発生しました: {e}", exc_info=True)
            return False
    
    def get_arm_position(self) -> Optional[Tuple[float, float, float]]:
        """
        アームの現在位置を取得
        
        Returns:
            エンドエフェクタの位置 (x, y, z)
        """
        if self.client_id is None or self.robot_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        try:
            link_state = p.getLinkState(
                self.robot_id,
                self.end_effector_index,
                physicsClientId=self.client_id
            )
            pos = link_state[4]  # 世界座標系での位置
            return pos
        except Exception as e:
            logger.error(f"❌ [PYBULLET_ENV] アーム位置取得中にエラーが発生しました: {e}", exc_info=True)
            return None
    
    def capture_image(self) -> 'Image':
        """
        カメラ画像をキャプチャ
        
        Returns:
            PIL Image
        """
        if self.client_id is None:
            raise RuntimeError("環境が作成されていません。")
        
        import numpy as np
        from PIL import Image
        
        view_matrix = p.computeViewMatrix([1.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.33, 0.1, 100.0)
        _, _, rgb_img, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix, physicsClientId=self.client_id)
        return Image.fromarray(np.reshape(rgb_img, (480, 640, 4))[:, :, :3].astype('uint8'))
    
    def cleanup(self):
        """環境をクリーンアップ"""
        if self.client_id is not None:
            try:
                p.disconnect(physicsClientId=self.client_id)
                logger.info(f"✅ [PYBULLET_ENV] 環境をクリーンアップしました (Client ID: {self.client_id})")
            except Exception as e:
                logger.error(f"❌ [PYBULLET_ENV] クリーンアップ中にエラーが発生しました: {e}", exc_info=True)
            finally:
                self.client_id = None
                self.robot_id = None
                self.object_ids = {}
                self.plane_id = None


# グローバルな環境インスタンス（ツール実行時に使用）
_global_env: Optional[PyBulletEnvironment] = None


def get_environment(use_gui: bool = False) -> PyBulletEnvironment:
    """
    新しいPyBullet環境を取得（毎回新しい環境を作成）
    
    Args:
        use_gui: GUIモードを使用するかどうか
    
    Returns:
        PyBulletEnvironmentインスタンス
    """
    global _global_env
    
    # 既存の環境があればクリーンアップ
    if _global_env is not None:
        _global_env.cleanup()
    
    # 新しい環境を作成
    _global_env = PyBulletEnvironment(use_gui=use_gui)
    _global_env.create_environment()
    
    return _global_env
