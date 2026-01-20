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

# Vision60ロボットをロード（学習時と同じ設定）
# 1. スポーン位置を低く設定（0.3m）
spawn_height = 0.3
robot_id = p.loadURDF("quadruped/vision60.urdf", [0, 0, spawn_height])
print("✅ Vision60ロボットをロードしました")

# 仕様書に基づくジョイントマッピング
joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
print(f"📊 可動ジョイント数: {len(joint_indices)} (Abduction, Hip, Knee × 4脚)")

# 2. 初期姿勢の設定（学習時と同じ）
knee_angle = 1.0
abd_angle = 0.2  # ハの字の角度

# Abductionジョイントの符号を個別に設定（check_vision.pyと同じ設定）
# i=0, j_idx=0: 左前（FL）
# i=3, j_idx=4: 右前（FR）
# i=6, j_idx=8: 左後ろ（RL）
# i=9, j_idx=12: 右後ろ（RR）
abd_signs = {
    0: 1.0,   # 左前（FL）: +1.0でプラス、-1.0でマイナス
    3: 1.0,   # 右前（FR）: +1.0でプラス、-1.0でマイナス
    6: -1.0,  # 左後ろ（RL）: +1.0でプラス、-1.0でマイナス
    9: -1.0,  # 右後ろ（RR）: +1.0でプラス、-1.0でマイナス
}

for i, j_idx in enumerate(joint_indices):
    if i in [0, 3, 6, 9]:  # Abductionジョイント (ハの字)
        init_pos = abd_angle * abd_signs[i]
        leg_names = {0: "FL", 3: "FR", 6: "RL", 9: "RR"}
        leg = leg_names[i]
    elif i in [2, 5, 8, 11]:  # Knee
        init_pos = knee_angle
        leg = None
    else:  # Hip
        init_pos = 0.0
        leg = None
    
    p.resetJointState(robot_id, j_idx, init_pos)
    # 初期状態で崩れないようモーターを保持（学習時と同じ）
    p.setJointMotorControl2(
        robot_id, j_idx, p.POSITION_CONTROL,
        targetPosition=init_pos, force=150.0,
        positionGain=0.05,   # 標準の半分にして「柔らかく」
        velocityGain=1.5      # 少し増やして「跳ね」を抑える
    )
    
    if i in [0, 3, 6, 9]:
        print(f"  初期姿勢: ジョイント {j_idx} (Abduction {leg}) を {init_pos:+.2f} rad に設定")
    elif i in [2, 5, 8, 11]:
        print(f"  初期姿勢: ジョイント {j_idx} (Knee) を {init_pos:.2f} rad に設定")

# 3. 安定待ちフェーズ（100ステップ、学習時と同じ）
print("⏳ 100ステップ安定を待ちます（ウォームアップ）...")
for _ in range(100):
    p.stepSimulation()
print("✅ ウォームアップ完了")

# --- 2. モデルの読み込み ---
model_path = "ppo_vision60_stable_v1"
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
    print(f"   先に train_vision_stable_v1.py を実行してモデルを学習してください")
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
    # トロット用の基準フェーズ（1.5Hz）
    phase = np.sin(2 * np.pi * 1.5 * (step_count * 0.01))
    return np.array([pos[2], vel[2]] + list(euler) + joint_angles + [phase], dtype=np.float32)

obs = get_obs()
print("✅ 初期観測値を取得しました")

print("\n📺 GUIで再生を開始します（Ctrl+Cで終了）...")
print("   Vision60ロボットの安定化トロット動作を表示します（stable_v1版）")
print("   📝 状態を vision60_flight_log_stable_v1.csv にログ記録します\n")
input("⏸️  Enterキーを押すと再生を開始します...")

# --- 4. ログファイルの準備 ---
log_filename = 'vision60_flight_log_stable_v1.csv'
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
        phase_a = np.sin(2 * np.pi * 1.5 * t)
        phase_b = -phase_a
        
        # 12つの関節すべてにアクションを適用（学習時と同じ：POSITION_CONTROL）
        for i, j_idx in enumerate(joint_indices):
            # 基本姿勢の維持（学習時と同じ）
            if i in [0, 3, 6, 9]:  # Abduction: ハの字保持
                # Abductionジョイントの符号を個別に設定（学習時と同じ）
                target_pos = 0.2 * abd_signs[i]
            elif i in [2, 8]:  # FR, RL knee
                target_pos = 1.0 + phase_a * 0.4
            elif i in [5, 11]:  # FL, RR knee
                target_pos = 1.0 + phase_b * 0.4
            else:  # Hip
                target_pos = 0.0
            
            # AIのアクションを加算（学習時と同じ）
            target_pos += action[i] * 0.2
            
            # positionGainを下げて「柔らかく」、velocityGainを上げて「粘り」を出す（学習時と同じ）
            p.setJointMotorControl2(
                robot_id, j_idx, p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=100.0,
                positionGain=0.05,   # 標準の半分にして「柔らかく」
                velocityGain=1.5      # 少し増やして「跳ね」を抑える
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
        # joint_indices[2]=2 (FL_knee), joint_indices[5]=6 (FR_knee), 
        # joint_indices[8]=10 (RL_knee), joint_indices[11]=14 (RR_knee)
        # FL, FR, RL, RR の順序で取得
        knee_joint_indices = [joint_indices[2], joint_indices[5], joint_indices[8], joint_indices[11]]  # FL, FR, RL, RR
        contacts = []
        
        for knee_joint_idx in knee_joint_indices:
            # 各Kneeジョイントの子リンク（足先）のインデックスを取得
            # PyBulletでは、通常ジョイントインデックス+1が子リンクインデックス
            child_link_index = knee_joint_idx + 1
            
            # そのリンクが地面（bodyB=-1）と接触しているか確認
            contact_points = p.getContactPoints(robot_id, -1, child_link_index)
            # 接触点があれば接地している
            has_contact = len(contact_points) > 0
            contacts.append(1 if has_contact else 0)
        
        # CSVに書き込み
        # Knee角度: joint_indices[2], [5], [8], [11] に対応（FL, FR, RL, RR）
        row = [
            step_count,
            pos[0], pos[2],  # pos_x, pos_z
            vel[0],  # vel_x
            euler[0], euler[1],  # roll, pitch
            joint_angles[2], joint_angles[5], joint_angles[8], joint_angles[11],  # FL, FR, RL, RR knee
            contacts[0], contacts[1], contacts[2], contacts[3]  # FL, FR, RL, RR contact
        ]
        writer.writerow(row)
