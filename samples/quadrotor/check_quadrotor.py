import pybullet as p
import pybullet_data
import numpy as np
import time

# --- 1. GUIモードで接続 ---
print("🚀 PyBulletをGUIモードで起動します...")
device_id = p.connect(p.GUI)
if device_id < 0:
    print("❌ GUIモードでの接続に失敗しました")
    exit(1)

# プロジェクトのQuadrotorディレクトリを検索パスに追加
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
quadrotor_dir = os.path.join(project_root, "Quadrotor")
data_path = pybullet_data.getDataPath()

# PyBulletのデフォルトデータパスとQuadrotorディレクトリの両方を追加
p.setAdditionalSearchPath(data_path)
p.setAdditionalSearchPath(quadrotor_dir)
print(f"📁 Quadrotorディレクトリを検索パスに追加: {quadrotor_dir}")

p.setGravity(0, 0, -9.81)

# plane.urdfを絶対パスでロード
plane_path = os.path.join(data_path, "plane.urdf")
p.loadURDF(plane_path)
print("✅ 床をロードしました")

# --- 2. Quadrotorロボットをロード ---
spawn_height = 1.0  # クアッドコプターは空中から開始
print(f"📦 Quadrotorロボットを高さ {spawn_height}m からスポーンします...")

# プロジェクトのQuadrotorディレクトリからの絶対パス
quadrotor_path = os.path.join(quadrotor_dir, "quadrotor.urdf")
print(f"📂 URDFファイルパス: {quadrotor_path}")

if not os.path.exists(quadrotor_path):
    print(f"❌ URDFファイルが見つかりません: {quadrotor_path}")
    p.disconnect()
    exit(1)

try:
    robot_id = p.loadURDF(quadrotor_path, [0, 0, spawn_height])
    print(f"✅ Quadrotorロボットをロードしました")
except Exception as e:
    print(f"❌ Quadrotorのロードに失敗しました: {e}")
    p.disconnect()
    exit(1)

# --- 3. ロボットの情報を解析 ---
print("\n" + "="*60)
print("📊 Quadrotorロボットの解析")
print("="*60)

# 基本情報
num_joints = p.getNumJoints(robot_id)
print(f"\n🔧 ジョイント数: {num_joints}")
print(f"   注意: このQuadrotorはforce_element（プロペラ）を使用しているため、")
print(f"   通常のジョイントはありません。プロペラはURDFのforce_elementで定義されています。")

# URDFファイルからforce_element（プロペラ）情報を解析
print(f"\n🚁 プロペラ情報（URDFから解析）:")
try:
    import xml.etree.ElementTree as ET
    tree = ET.parse(quadrotor_path)
    root = tree.getroot()
    
    # 名前空間を処理
    ns = {'urdf': 'http://drake.mit.edu'}
    
    # 名前空間付きで検索
    propellers = root.findall('.//urdf:force_element', ns)
    
    for prop in propellers:
        prop_name = prop.get('name', 'unknown')
        propellor = prop.find('urdf:propellor', ns)
        if propellor is not None:
            lower_limit = propellor.get('lower_limit', 'N/A')
            upper_limit = propellor.get('upper_limit', 'N/A')
            scale_thrust = propellor.get('scale_factor_thrust', 'N/A')
            scale_moment = propellor.get('scale_factor_moment', 'N/A')
            
            origin = propellor.find('urdf:origin', ns)
            origin_xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
            
            axis = propellor.find('urdf:axis', ns)
            axis_xyz = axis.get('xyz', '0 0 1') if axis is not None else '0 0 1'
            
            print(f"  {prop_name}:")
            print(f"    位置: {origin_xyz}")
            print(f"    軸: {axis_xyz}")
            print(f"    回転範囲: [{lower_limit}, {upper_limit}]")
            print(f"    推力スケール: {scale_thrust}")
            print(f"    モーメントスケール: {scale_moment}")
except Exception as e:
    import traceback
    print(f"  ⚠️ プロペラ情報の解析に失敗しました: {e}")
    print(f"  詳細: {traceback.format_exc()}")

# ベース情報
base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
base_vel, base_ang_vel = p.getBaseVelocity(robot_id)
print(f"\n📍 ベース位置: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
print(f"📍 ベース姿勢 (Quaternion): ({base_orn[0]:.3f}, {base_orn[1]:.3f}, {base_orn[2]:.3f}, {base_orn[3]:.3f})")
euler = p.getEulerFromQuaternion(base_orn)
print(f"📍 ベース姿勢 (Euler): Roll={euler[0]:.3f}rad, Pitch={euler[1]:.3f}rad, Yaw={euler[2]:.3f}rad")
print(f"📍 ベース速度: ({base_vel[0]:.3f}, {base_vel[1]:.3f}, {base_vel[2]:.3f}) m/s")
print(f"📍 ベース角速度: ({base_ang_vel[0]:.3f}, {base_ang_vel[1]:.3f}, {base_ang_vel[2]:.3f}) rad/s")

# ジョイント情報
if num_joints > 0:
    print(f"\n🔩 ジョイント詳細:")
    actuable_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_index = joint_info[0]
        joint_name = joint_info[1].decode('utf-8') if joint_info[1] else f"joint_{i}"
        joint_type = joint_info[2]
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]
        joint_max_force = joint_info[10]
        joint_max_velocity = joint_info[11]
        
        # ジョイントタイプの説明
        joint_type_names = {
            p.JOINT_REVOLUTE: "Revolute (回転)",
            p.JOINT_PRISMATIC: "Prismatic (直動)",
            p.JOINT_SPHERICAL: "Spherical (球)",
            p.JOINT_PLANAR: "Planar (平面)",
            p.JOINT_FIXED: "Fixed (固定)"
        }
        joint_type_name = joint_type_names.get(joint_type, f"Unknown ({joint_type})")
        
        # 現在のジョイント状態
        joint_state = p.getJointState(robot_id, i)
        joint_angle = joint_state[0]
        joint_velocity = joint_state[1]
        
        print(f"\n  ジョイント {i}:")
        print(f"    名前: {joint_name}")
        print(f"    タイプ: {joint_type_name}")
        print(f"    現在角度: {joint_angle:.3f} rad")
        print(f"    現在速度: {joint_velocity:.3f} rad/s")
        
        if joint_type != p.JOINT_FIXED:
            print(f"    可動範囲: [{joint_lower_limit:.3f}, {joint_upper_limit:.3f}] rad")
            print(f"    最大トルク: {joint_max_force:.1f} N⋅m")
            print(f"    最大速度: {joint_max_velocity:.3f} rad/s")
            actuable_joints.append(i)
        else:
            print(f"    (固定ジョイント)")

    print(f"\n✅ 可動ジョイント数: {len(actuable_joints)}")
    print(f"   インデックス: {actuable_joints}")
else:
    print(f"\n🔩 ジョイント詳細: なし（force_elementを使用）")
    actuable_joints = []

# リンク情報
print(f"\n🔗 リンク情報:")
for i in range(-1, num_joints):  # -1はベースリンク
    if i == -1:
        link_name = "base_link"
        link_pos, link_orn = p.getBasePositionAndOrientation(robot_id)
    else:
        link_info = p.getLinkState(robot_id, i)
        link_name = p.getJointInfo(robot_id, i)[12].decode('utf-8') if p.getJointInfo(robot_id, i)[12] else f"link_{i}"
        link_pos = link_info[0]
        link_orn = link_info[1]
    
    print(f"  リンク {i} ({link_name}):")
    print(f"    位置: ({link_pos[0]:.3f}, {link_pos[1]:.3f}, {link_pos[2]:.3f})")
    euler = p.getEulerFromQuaternion(link_orn)
    print(f"    姿勢: Roll={euler[0]:.3f}, Pitch={euler[1]:.3f}, Yaw={euler[2]:.3f}")

# --- 4. 質量と慣性の情報 ---
print(f"\n⚖️ 質量・慣性情報:")
dyn_info = p.getDynamicsInfo(robot_id, -1)
base_mass = dyn_info[0]
base_friction = dyn_info[1]
base_inertia = dyn_info[2]  # 慣性テンソル（タプル）
base_restitution = dyn_info[5]  # 反発係数
print(f"  ベース質量: {base_mass:.3f} kg")
print(f"  ベース摩擦係数: {base_friction:.3f}")
print(f"  ベース慣性テンソル: ({base_inertia[0]:.6f}, {base_inertia[1]:.6f}, {base_inertia[2]:.6f})")
print(f"  ベース反発係数: {base_restitution:.3f}")

for i in range(num_joints):
    dyn_info = p.getDynamicsInfo(robot_id, i)
    mass = dyn_info[0]
    if mass > 0:
        joint_name = p.getJointInfo(robot_id, i)[1].decode('utf-8') if p.getJointInfo(robot_id, i)[1] else f"joint_{i}"
        print(f"  ジョイント {i} ({joint_name}) 質量: {mass:.3f} kg")

# --- 5. 簡単な動作テスト ---
print(f"\n" + "="*60)
print("🧪 動作テスト")
print("="*60)

if len(actuable_joints) > 0:
    print(f"\n🔧 可動ジョイントに初期トルクを設定します...")
    for j_idx in actuable_joints:
        joint_info = p.getJointInfo(robot_id, j_idx)
        joint_name = joint_info[1].decode('utf-8') if joint_info[1] else f"joint_{j_idx}"
        
        # クアッドコプターの場合、通常はプロペラの回転速度を制御
        # ここでは位置制御で初期姿勢を保持
        p.setJointMotorControl2(
            robot_id, j_idx, p.POSITION_CONTROL,
            targetPosition=0.0, force=10.0
        )
        print(f"  ジョイント {j_idx} ({joint_name}): 位置制御を設定 (目標: 0.0 rad)")
else:
    print(f"\n💡 このQuadrotorはforce_elementを使用しているため、")
    print(f"   プロペラの制御はapplyExternalForceやカスタムコントローラーで行います。")

# --- 6. シミュレーションを実行して状態を確認 ---
test_steps = 100
print(f"\n⏳ {test_steps}ステップシミュレーションを実行します...")
print("   GUIでロボットの状態を確認してください\n")

height_history = []
roll_history = []
pitch_history = []
yaw_history = []

for step in range(test_steps):
    # 物理シミュレーションを1ステップ進める
    p.stepSimulation()
    
    # 状態を取得
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    vel, ang_vel = p.getBaseVelocity(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    
    height = pos[2]
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    
    # 状態を記録
    height_history.append(height)
    roll_history.append(roll)
    pitch_history.append(pitch)
    yaw_history.append(yaw)
    
    # 一定間隔で状態を表示
    if step % 20 == 0 or step < 5:
        print(f"Step {step:3d}: 高さ={height:.3f}m, Roll={roll:.3f}rad, Pitch={pitch:.3f}rad, Yaw={yaw:.3f}rad")
        print(f"           速度=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) m/s")
    
    # リアルタイム表示のため少し待つ
    time.sleep(1.0 / 240.0)

# --- 7. 結果の確認と表示 ---
print("\n" + "="*60)
print("📊 テスト結果")
print("="*60)

final_height = height_history[-1]
final_roll = roll_history[-1]
final_pitch = pitch_history[-1]
final_yaw = yaw_history[-1]

avg_height = np.mean(height_history)
avg_roll = np.mean(np.abs(roll_history))
avg_pitch = np.mean(np.abs(pitch_history))
avg_yaw = np.mean(np.abs(yaw_history))

print(f"\n最終状態:")
print(f"  高さ: {final_height:.3f}m")
print(f"  Roll: {final_roll:.3f}rad ({np.degrees(final_roll):.1f}度)")
print(f"  Pitch: {final_pitch:.3f}rad ({np.degrees(final_pitch):.1f}度)")
print(f"  Yaw: {final_yaw:.3f}rad ({np.degrees(final_yaw):.1f}度)")

print(f"\n平均状態:")
print(f"  平均高さ: {avg_height:.3f}m")
print(f"  平均Roll絶対値: {avg_roll:.3f}rad ({np.degrees(avg_roll):.1f}度)")
print(f"  平均Pitch絶対値: {avg_pitch:.3f}rad ({np.degrees(avg_pitch):.1f}度)")
print(f"  平均Yaw絶対値: {avg_yaw:.3f}rad ({np.degrees(avg_yaw):.1f}度)")

print(f"\n💡 次のステップ:")
print(f"   - プロペラの回転速度を制御してホバリングを実現")
print(f"   - PID制御で姿勢を安定化")
print(f"   - 目標位置への移動制御を実装")

print(f"\n📝 注意:")
print(f"   - プロペラ制御がないため、Quadrotorは重力で落下します（正常な動作です）")
print(f"   - プロペラの制御を実装すると、ホバリングや飛行が可能になります")

# GUIモードの場合は待機、DIRECTモードの場合は即座に終了
if device_id >= 0 and p.getConnectionInfo(device_id)['connectionMethod'] == p.GUI:
    print("\n⏸️  Enterキーを押すと終了します...")
    try:
        input()
    except:
        pass

p.disconnect()
print("✅ シミュレーションを終了しました")
