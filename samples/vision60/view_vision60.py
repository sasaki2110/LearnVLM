import pybullet as p
import pybullet_data
import numpy as np
import time  # リアルタイム再生の速度調整用
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

# --- 2. モデルの読み込み ---
model_path = "ppo_vision60_position_step"
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
    print(f"   先に train_vision60.py を実行してモデルを学習してください")
    exit(1)

# --- 3. 初期観測値を取得 ---
def get_obs():
    """観測値を取得（学習時と同じ形式）"""
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    vel, _ = p.getBaseVelocity(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    joint_angles = [p.getJointState(robot_id, i)[0] for i in joint_indices]
    return np.array([pos[2], vel[2]] + list(euler) + joint_angles, dtype=np.float32)

obs = get_obs()
print("✅ 初期観測値を取得しました")

print("\n📺 GUIで再生を開始します（Ctrl+Cで終了）...")
print("   Vision60ロボットの動作を表示します\n")
input("⏸️  Enterキーを押すと再生を開始します...")

# --- 4. 実行ループ ---
while True:
    # 物理演算の1ステップあたりの時間を考慮して少し待つ（これがないと超高速で終わります）
    time.sleep(1.0 / 240.0)
    
    # アクションを予測
    action, _ = model.predict(obs, deterministic=True)
    
    # 12つの関節すべてにアクションを適用（学習時と同じ：POSITION_CONTROL）
    for i, j_idx in enumerate(joint_indices):
        if i in [0, 3, 6, 9]:  # Abduction (±0.43 rad)
            target_pos = action[i] * 0.43
            force = 300.0
        elif i in [1, 4, 7, 10]:  # Hip (±3.14 rad)
            target_pos = action[i] * 3.14
            force = 80.0
        else:  # Knee (0 ~ 3.14 rad)
            # action[-1, 1] -> [0, 3.14]
            target_pos = (action[i] + 1) * 1.57
            force = 80.0
        
        p.setJointMotorControl2(
            robot_id, j_idx, p.POSITION_CONTROL,
            targetPosition=target_pos, force=force
        )
    
    p.stepSimulation()
    
    # 次の状態取得
    obs = get_obs()
    
    # 転倒リセット（学習時と同じ終了判定）
    height = obs[0]
    roll, pitch = obs[2], obs[3]
    
    """
    if height < 0.3 or abs(roll) > 0.5 or abs(pitch) > 0.5:
        print(f"⚠️  転倒を検出。リセットします... (高さ: {height:.3f}, roll: {roll:.3f}, pitch: {pitch:.3f})")
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], [0, 0, 0, 1])
        # ジョイントもリセット（初期姿勢に戻す）
        for i, j_idx in enumerate(joint_indices):
            if i in [2, 5, 8, 11]:  # 膝を少し曲げる
                p.resetJointState(robot_id, j_idx, 1.0)
            else:
                p.resetJointState(robot_id, j_idx, 0, 0)
        obs = get_obs()
    """