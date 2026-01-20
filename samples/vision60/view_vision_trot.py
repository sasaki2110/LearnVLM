import pybullet as p
import pybullet_data
import numpy as np
import time  # リアルタイム再生の速度調整用
import csv  # ロギング用
from stable_baselines3 import PPO

# --- 1. GUIモードで接続 ---
print("🚀 PyBulletをGUIモードで起動します...")
device_id = p.connect(p.GUI)
if device_id < 0:
    print("❌ GUIモードでの接続に失敗しました")
    exit(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
print("✅ 床をロードしました")

# Vision60ロボットをロード（学習時と同じ高さから開始）
robot_id = p.loadURDF("quadruped/vision60.urdf", [0, 0, 0.5])
print("✅ Vision60ロボットをロードしました")

# 仕様書に基づくジョイントマッピング
joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
print(f"📊 可動ジョイント数: {len(joint_indices)} (Abduction, Hip, Knee × 4脚)")

# 初期姿勢: 少し膝を曲げておくと立ち上がりやすい（学習時と同じ）
for i, j_idx in enumerate(joint_indices):
    # 膝(Knee)はIndex 2, 5, 8, 11 (リスト内では 2, 5, 8, 11番目)
    if i in [2, 5, 8, 11]:
        p.resetJointState(robot_id, j_idx, 1.0)
        print(f"  初期姿勢: ジョイント {j_idx} (Knee) を 1.0 rad に設定")
    else:
        p.resetJointState(robot_id, j_idx, 0.0)

# --- 2. モデルの読み込み ---
model_path = "ppo_vision60_trot_base"
try:
    # .zip拡張子を試す
    try:
        model = PPO.load(f"{model_path}.zip")
        print(f"✅ モデル '{model_path}.zip' を読み込みました")
    except FileNotFoundError:
        # .zipなしで試す
        model = PPO.load(model_path)
        print(f"✅ モデル '{model_path}' を読み込みました")
except FileNotFoundError:
    print(f"❌ エラー: モデルファイル '{model_path}' または '{model_path}.zip' が見つかりません")
    print(f"   先に train_vision_trot.py を実行してモデルを学習してください")
    exit(1)

# --- 3. 初期観測値を取得（学習時と同じ形式） ---
step_count = 0

def get_obs():
    """観測値を取得（学習時と同じ形式：18次元）"""
    global step_count
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    vel, _ = p.getBaseVelocity(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    joint_angles = [p.getJointState(robot_id, i)[0] for i in joint_indices]
    
    # トロットのリズム（サイン波）を観測に加える（学習時と同じ）
    phase = np.sin(2 * np.pi * 1.5 * (step_count * 0.01))
    
    return np.array([pos[2], vel[2]] + list(euler) + joint_angles + [phase], dtype=np.float32)

obs = get_obs()
print("✅ 初期観測値を取得しました")

print("\n📺 GUIで再生を開始します（Ctrl+Cで終了）...")
print("   Vision60ロボットのトロット（足踏み）動作を表示します")
print("   📹 カメラがロボットを自動追跡します")
print("   📝 状態を vision60_flight_log.csv にログ記録します\n")
input("⏸️  Enterキーを押すと再生を開始します...")

# --- 4. ログファイルの準備 ---
log_filename = 'vision60_flight_log.csv'
with open(log_filename, mode='w', newline='') as log_file:
    writer = csv.writer(log_file)
    # ヘッダー作成
    header = ['step', 'pos_x', 'pos_z', 'vel_x', 'roll', 'pitch', 
              'FL_knee', 'FR_knee', 'RL_knee', 'RR_knee', 
              'FL_contact', 'FR_contact', 'RL_contact', 'RR_contact']
    writer.writerow(header)
    print(f"✅ ログファイル '{log_filename}' を作成しました")

# --- 5. カメラ設定 ---
# カメラの初期設定（ロボットを追跡するように設定）
camera_distance = 2.0  # カメラからロボットまでの距離
camera_yaw = 45.0      # 水平方向の角度（度）
camera_pitch = -20.0   # 垂直方向の角度（度、下向きが負）

# --- 6. 実行ループ ---
# ログファイルを追記モードで開く
with open(log_filename, mode='a', newline='') as log_file:
    writer = csv.writer(log_file)
    
    while True:
        # 物理演算の1ステップあたりの時間を考慮して少し待つ（これがないと超高速で終わります）
        time.sleep(1.0 / 240.0)
        
        # アクションを予測
        action, _ = model.predict(obs, deterministic=True)
        
        # 学習時と同じトロット制御ロジックを適用
        t = step_count * 0.01
        # 1.5Hzのリズムで対角線の足を交互に
        phase_a = np.sin(2 * np.pi * 1.5 * t)
        phase_b = -phase_a
        
        # 12つの関節すべてにアクションを適用（学習時と同じ：POSITION_CONTROL）
        for i, j_idx in enumerate(joint_indices):
            # 基準となるトロットの動きを計算（Kneeをメインに動かす）
            target_pos = 0.0
            if i in [2, 8]:  # FR, RL の Knee
                target_pos = 1.0 + phase_a * 0.5
            elif i in [5, 11]:  # FL, RR の Knee
                target_pos = 1.0 + phase_b * 0.5
            
            # AIのアクションを「補正値」として加える
            target_pos += action[i] * 0.2
            
            # 可動範囲制限（Knee: 0~3.14）
            target_pos = np.clip(target_pos, 0, 3.1)
            
            p.setJointMotorControl2(
                robot_id, j_idx, p.POSITION_CONTROL,
                targetPosition=target_pos, force=150.0
            )
        
        p.stepSimulation()
        step_count += 1
        
        # カメラをロボットの位置に追跡させる
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=robot_pos
        )
        
        # 次の状態取得
        obs = get_obs()
        
        # ログ用のデータを取得
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(orn)
        joint_angles = [p.getJointState(robot_id, i)[0] for i in joint_indices]
        
        # 接地情報の取得（各脚の足先リンクの接地判定）
        # Vision60の各脚の足先リンクインデックス
        # 各Kneeジョイントの子リンクが足先リンク（通常はジョイントインデックス+1）
        # joint_indices[2]=2 (FR_knee), joint_indices[5]=5 (FL_knee), 
        # joint_indices[8]=8 (RL_knee), joint_indices[11]=14 (RR_knee)
        knee_joint_indices = [joint_indices[2], joint_indices[5], joint_indices[8], joint_indices[11]]
        contacts = []
        
        for knee_joint_idx in knee_joint_indices:
            # 各Kneeジョイントの子リンク（足先）のインデックスを取得
            # PyBulletでは、通常ジョイントインデックス+1が子リンクインデックス
            # ただし、URDFの構造により異なる場合があるため、複数の方法を試す
            child_link_index = knee_joint_idx + 1
            
            # そのリンクが地面（リンクインデックス-1）と接触しているか確認
            # p.getContactPoints(bodyA, bodyB, linkIndexA, linkIndexB)
            contact_points = p.getContactPoints(robot_id, -1, child_link_index)
            # 接触点があれば接地している
            has_contact = len(contact_points) > 0
            contacts.append(1 if has_contact else 0)
        
        # データの書き込み
        # Knee角度: joint_indices[2], [5], [8], [11] に対応
        # 提供されたコードの形式に合わせて簡易版
        writer.writerow([
            step_count,
            pos[0], pos[2],  # 位置（x, z）
            vel[0],  # 速度（x）
            euler[0], euler[1],  # 姿勢（roll, pitch）
            joint_angles[2], joint_angles[5], joint_angles[8], joint_angles[11],  # Knee角度
            *contacts  # 接地情報
        ])
        
        # 転倒リセット（学習時と同じ終了判定）
        height = obs[0]
        roll, pitch = obs[2], obs[3]
    
    """
    if height < 0.3 or abs(roll) > 0.6 or abs(pitch) > 0.6:
        print(f"⚠️  転倒を検出。リセットします... (高さ: {height:.3f}, roll: {roll:.3f}, pitch: {pitch:.3f})")
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], [0, 0, 0, 1])
        # ジョイントもリセット（初期姿勢に戻す）
        for i, j_idx in enumerate(joint_indices):
            if i in [2, 5, 8, 11]:  # 膝を少し曲げる
                p.resetJointState(robot_id, j_idx, 1.0)
            else:
                p.resetJointState(robot_id, j_idx, 0, 0)
        step_count = 0
        obs = get_obs()
    """