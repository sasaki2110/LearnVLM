import pybullet as p
import pybullet_data
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time

# --- 1. 環境とVLMの初期化 ---
#p.connect(p.DIRECT)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
# Franka Pandaをロード（グリッパー付き）
try:
    arm_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    print("✅ Franka Pandaをロードしました")
except:
    # パスが異なる場合のフォールバック
    try:
        arm_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        print("✅ Franka Pandaをロードしました（フォールバック）")
    except:
        print("❌ Franka PandaのURDFが見つかりません。kuka_iiwaを使用します。")
        arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# アヒルを配置
duck_target_pos = [0.6, 0.3, 0.0] 
duck_id = p.loadURDF("duck_vhacd.urdf", basePosition=duck_target_pos)

# シミュレーションを少し進める（安定化のため）
for _ in range(500):
    p.stepSimulation()
    #time.sleep(1./240.)

# アームの関節情報を確認
num_joints = p.getNumJoints(arm_id)
print(f"📊 アームの関節数: {num_joints}")

# グリッパーの関節を探す
gripper_joints = []
for i in range(num_joints):
    joint_info = p.getJointInfo(arm_id, i)
    joint_name = joint_info[1].decode('utf-8')
    if 'finger' in joint_name.lower() or 'gripper' in joint_name.lower():
        gripper_joints.append(i)
        print(f"🤏 グリッパー関節発見: {joint_name} (インデックス: {i})")

# エンドエフェクタのリンクインデックス（通常は最後のリンク）
ee_link_index = num_joints - 1
print(f"📍 エンドエフェクタリンクインデックス: {ee_link_index}")

# VLMロード前にアヒルの位置をロギング
duck_pos, duck_orn = p.getBasePositionAndOrientation(duck_id)
print(f"📍 VLMロード前のアヒル位置: [{duck_pos[0]:.4f}, {duck_pos[1]:.4f}, {duck_pos[2]:.4f}]")
print(f"   初期設定位置: [{duck_target_pos[0]:.4f}, {duck_target_pos[1]:.4f}, {duck_target_pos[2]:.4f}]")

print("🚀 VLMロード中...")
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- 2. 視覚取得関数 ---
def capture_image():
    view_matrix = p.computeViewMatrix([1.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.33, 0.1, 100.0)
    _, _, rgb_img, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix)
    return Image.fromarray(np.reshape(rgb_img, (480, 640, 4))[:, :, :3].astype('uint8'))

# --- 3. 座標変換関数（既存と同じ） ---
def map_vlm_to_world(vlm_coords, camera_eye=[1.0, 0.0, 1.0], camera_target=[0.5, 0.0, 0.0], 
                     fov=60, image_size=(640, 480), plane_z=0.0, **kwargs):
    y_center = (vlm_coords[0] + vlm_coords[2]) / 2
    x_center = (vlm_coords[1] + vlm_coords[3]) / 2
    bbox_width = vlm_coords[3] - vlm_coords[1]
    bbox_height = vlm_coords[2] - vlm_coords[0]
    bbox_size = max(bbox_width, bbox_height)
    distance_factor = kwargs.get('distance_factor', 1.0)
    world_x = 0.8 - (y_center * 0.4 * distance_factor)
    world_y = 0.4 - (x_center * 0.5 * distance_factor)
    estimated_z = plane_z + 0.016 + (1.0 - bbox_size) * 0.02
    return [float(world_x), float(world_y), float(estimated_z)]

# --- 4. アーム制御関数 ---
def move_arm_to(target_pos):
    """アームを目標位置に移動"""
    # 逆運動学で各関節の目標角度を計算
    joint_poses = p.calculateInverseKinematics(arm_id, ee_link_index, target_pos)
    for i in range(len(joint_poses)):
        p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_poses[i])
    for _ in range(200): 
        p.stepSimulation()
        #time.sleep(1./240.)

def get_arm_position():
    """現在のアーム先端位置を取得"""
    arm_ee_state = p.getLinkState(arm_id, ee_link_index)
    return arm_ee_state[0]

def control_gripper(open_width=0.04, force=20.0):
    """
    グリッパーを制御
    open_width: 開く幅（m）。0.04 = 4cm開く、0.0 = 閉じる
    force: グリッパーの力（N）
    """
    if len(gripper_joints) == 0:
        print("⚠️ グリッパー関節が見つかりません")
        return
    
    # 各グリッパー関節を制御
    for joint_idx in gripper_joints:
        # 開く幅を各指に分配（通常は2本の指がある）
        finger_position = open_width / 2.0
        p.setJointMotorControl2(
            arm_id, 
            joint_idx, 
            p.POSITION_CONTROL, 
            targetPosition=finger_position,
            force=force
        )
    
    # グリッパーの動作を反映
    for _ in range(200):  # より長く動作を反映
        p.stepSimulation()
        #time.sleep(1./240.)

def detect_duck_position(show_debug=False, current_arm_pos=None):
    """VLMを使ってアヒルの位置を検出"""
    img = capture_image()
    enc_image = model.encode_image(img)
    coords = model.answer_question(enc_image, "Point out the duck with a bounding box.", tokenizer)
    
    try:
        num_coords = eval(coords)
        bbox_size = max(num_coords[3] - num_coords[1], num_coords[2] - num_coords[0])
        confidence = min(bbox_size * 2.0, 1.0)
        
        if current_arm_pos is not None:
            arm_to_target = np.linalg.norm(np.array(current_arm_pos[:2]) - np.array([0.5, 0.0]))
            distance_factor = max(0.5, min(1.5, arm_to_target / 0.5))
        else:
            distance_factor = 1.0
        
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
            print(f"🔍 マッピング詳細: y_center={y_center:.3f}, x_center={x_center:.3f}, "
                  f"bbox_size={bbox_size:.3f}, confidence={confidence:.3f}, "
                  f"distance_factor={distance_factor:.3f}, world=[{target_3d[0]:.3f}, {target_3d[1]:.3f}, {target_3d[2]:.3f}]")
        
        estimated_distance = 0.5 / (bbox_size + 0.01)
        estimated_distance = max(0.05, min(1.0, estimated_distance))
        
        return target_3d, num_coords, confidence, bbox_size, estimated_distance
    except:
        print(f"❌ 座標がうまく取れませんでした: {coords}")
        return None, None, 0.0, 0.0, 1.0

# --- 5. 段階的な接近アプローチ（グリッパー制御付き） ---
print("=" * 60)
print("🎯 段階的な接近アプローチ開始（Franka Panda + グリッパー）")
print("=" * 60)

# グリッパーを開く
print("\n🤏 グリッパーを開きます...")
control_gripper(open_width=0.04)
time.sleep(0.5)

# ステップ1: VLMで目標位置を決定
print("\n📸 ステップ1: VLMで目標位置を決定")
current_pos = get_arm_position()
result = detect_duck_position(show_debug=True, current_arm_pos=current_pos)
if result[0] is None:
    print("❌ 目標位置の検出に失敗しました")
    exit(1)

target_3d, num_coords, confidence, _, _ = result
print(f"🎯 目標位置: [{target_3d[0]:.4f}, {target_3d[1]:.4f}, {target_3d[2]:.4f}]")

approach_height = 0.10
target_above = [target_3d[0], target_3d[1], target_3d[2] + approach_height]
print(f"📍 接近位置（目標の{approach_height*100:.0f}cm上）: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")

# ステップ2: 目標位置の10cm上まで段階的に移動
print(f"\n📈 ステップ2: 目標位置の{approach_height*100:.0f}cm上まで段階的に移動（10段階）")
num_steps_approach = 10

for step in range(num_steps_approach):
    print(f"\n--- 段階 {step + 1}/{num_steps_approach} ---")
    current_pos = get_arm_position()
    result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
    if result[0] is None:
        print("⚠️ 検出失敗、前回の目標位置を使用")
    else:
        new_target, _, _, _, _ = result
        target_above[0] = new_target[0]
        target_above[1] = new_target[1]
        print(f"🔄 目標位置を更新: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
    
    progress = (step + 1) / num_steps_approach
    intermediate_pos = [
        current_pos[i] + (target_above[i] - current_pos[i]) * progress
        for i in range(3)
    ]
    
    print(f"📍 現在位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    print(f"🎯 中間目標: [{intermediate_pos[0]:.4f}, {intermediate_pos[1]:.4f}, {intermediate_pos[2]:.4f}]")
    move_arm_to(intermediate_pos)
    time.sleep(0.3)

# ステップ3: 真上に移動するまでループ（改善版：重み付き平均を使用）
print(f"\n🎯 ステップ3: 真上に移動するまでループ（精度向上版）")
max_align_iterations = 20  # 反復回数を増やす
align_threshold = 0.015  # 閾値を厳しく（1.5cm）

# 過去の検出結果を保持（重み付き平均用）
detected_targets = []  # [(target, confidence), ...]

for align_iter in range(max_align_iterations):
    print(f"\n--- 位置合わせ {align_iter + 1}/{max_align_iterations} ---")
    current_pos = get_arm_position()
    result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
    if result[0] is None:
        print("❌ 検出失敗、終了します")
        break
    
    new_target, _, confidence, _, _ = result
    detected_targets.append((new_target, confidence))
    
    # 重み付き平均を計算（最新の検出ほど重みが大きい）
    if len(detected_targets) > 1:
        # 最新3回の検出を使用（重みは1, 2, 3）
        recent_targets = detected_targets[-3:]
        weights = list(range(1, len(recent_targets) + 1))
        total_weight = sum(weights)
        weighted_target = [
            sum(pos[i] * w for (pos, _), w in zip(recent_targets, weights)) / total_weight
            for i in range(3)
        ]
        print(f"📊 重み付き平均位置: [{weighted_target[0]:.4f}, {weighted_target[1]:.4f}, {weighted_target[2]:.4f}]")
        new_target = weighted_target
    
    xy_error = np.sqrt((current_pos[0] - new_target[0])**2 + (current_pos[1] - new_target[1])**2)
    print(f"📏 X, Y方向の誤差: {xy_error:.4f}m")
    
    if xy_error < align_threshold:
        print(f"✅ 真上に到達しました！（誤差: {xy_error:.4f}m < {align_threshold:.3f}m）")
        break
    
    target_above[0] = new_target[0]
    target_above[1] = new_target[1]
    
    # 移動ステップを小さくする（反復回数が増えるほど小さく）
    move_ratio = max(0.3, 0.7 - align_iter * 0.02)  # 最初0.7、徐々に小さく
    intermediate_pos = [
        current_pos[i] + (target_above[i] - current_pos[i]) * move_ratio
        for i in range(3)
    ]
    intermediate_pos[2] = target_above[2]
    
    print(f"🎯 目標位置: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
    print(f"📐 移動比率: {move_ratio:.2f}")
    move_arm_to(intermediate_pos)
    time.sleep(0.3)

# ステップ4: 少しずつ下がる（X, Yは固定、Zのみ下げる）
print(f"\n📉 ステップ4: 少しずつ下がる（X, Yは固定、Zのみ下げる）")
num_steps_descend = 10

# 真上に到達した位置を固定（X, Yは変更しない）
current_pos = get_arm_position()
fixed_target_xy = [current_pos[0], current_pos[1]]  # X, Yを固定
print(f"📍 固定されたX, Y位置: [{fixed_target_xy[0]:.4f}, {fixed_target_xy[1]:.4f}]")

start_z = current_pos[2]

# 最初にアヒルの位置を確認（安全マージンを決定）
result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
if result[0] is not None:
    duck_target, _, _, bbox_size, estimated_distance = result
    # バウンディングボックスが大きい（0.5以上）= 近い = 安全マージンを小さく
    if bbox_size > 0.5:
        safety_margin = 0.02  # 2cm
    else:
        safety_margin = max(0.02, min(0.05, estimated_distance * 0.1))
    end_z = duck_target[2] + safety_margin
    print(f"📏 アヒル位置: [{duck_target[0]:.4f}, {duck_target[1]:.4f}, {duck_target[2]:.4f}]")
    print(f"📏 バウンディングボックスサイズ: {bbox_size:.3f}")
    print(f"🛡️ 安全マージン: {safety_margin*100:.1f}cm, 停止位置Z: {end_z:.4f}m")
else:
    safety_margin = 0.03
    end_z = target_3d[2] + safety_margin
    print(f"⚠️ 検出失敗、デフォルトの安全マージン（3cm）を使用")

# Z方向のみを段階的に下げる
for step in range(num_steps_descend):
    print(f"\n--- 下降段階 {step + 1}/{num_steps_descend} ---")
    current_pos = get_arm_position()
    
    # X, Yは固定、Zのみを下げる
    progress = (step + 1) / num_steps_descend
    target_z = start_z - (start_z - end_z) * progress
    intermediate_pos = [fixed_target_xy[0], fixed_target_xy[1], target_z]
    
    print(f"📍 現在位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    print(f"🎯 中間目標: [{intermediate_pos[0]:.4f}, {intermediate_pos[1]:.4f}, {intermediate_pos[2]:.4f}]")
    
    if target_z < end_z:
        target_z = end_z
        intermediate_pos[2] = end_z
        print(f"⚠️ 安全マージンを維持: Z={end_z:.4f}m")
    
    move_arm_to(intermediate_pos)
    time.sleep(0.3)

# ステップ4.5: グリッパーで掴める位置までさらに下がる
print(f"\n📉 ステップ4.5: グリッパーで掴める位置までさらに下がる")
current_pos = get_arm_position()

# アヒルの実際の位置を再確認
result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
if result[0] is not None:
    duck_target, _, _, bbox_size, _ = result
    # アヒルの高さを考慮して、さらに1cm下がる
    final_grasp_z = duck_target[2] + 0.01  # 1cm上
    print(f"🦆 アヒル位置: [{duck_target[0]:.4f}, {duck_target[1]:.4f}, {duck_target[2]:.4f}]")
    print(f"⬇️ さらに下がります: {current_pos[2]:.4f}m → {final_grasp_z:.4f}m")
    
    # X, Yは固定、Zのみを下げる
    grasp_pos = [fixed_target_xy[0], fixed_target_xy[1], final_grasp_z]
    move_arm_to(grasp_pos)
    time.sleep(0.5)
    
    # 最終位置を確認
    final_pos_before_grasp = get_arm_position()
    print(f"📍 グリッパーを閉じる前の位置: [{final_pos_before_grasp[0]:.4f}, {final_pos_before_grasp[1]:.4f}, {final_pos_before_grasp[2]:.4f}]")
else:
    print(f"⚠️ 検出失敗、現在位置を維持: {current_pos[2]:.4f}m")

# ステップ5: グリッパーで掴む
print(f"\n🤏 ステップ5: グリッパーで掴む")

# グリッパーを閉じる前のアヒル位置を記録
duck_pos_before, _ = p.getBasePositionAndOrientation(duck_id)
print(f"🦆 掴む前のアヒル位置: [{duck_pos_before[0]:.4f}, {duck_pos_before[1]:.4f}, {duck_pos_before[2]:.4f}]")

print("🤏 グリッパーを閉じます...")
# グリッパーを閉じる（力は強めに）
control_gripper(open_width=0.0, force=50.0)  # 閉じる、力は50N
time.sleep(1.0)  # 掴む時間を確保

# アヒルが掴まれているか確認
duck_pos_after, _ = p.getBasePositionAndOrientation(duck_id)
print(f"🦆 掴んだ後のアヒル位置: [{duck_pos_after[0]:.4f}, {duck_pos_after[1]:.4f}, {duck_pos_after[2]:.4f}]")

# アヒルが動いたか確認（掴まれていれば位置が変わる可能性がある）
duck_movement = np.sqrt(sum([(duck_pos_after[i] - duck_pos_before[i])**2 for i in range(3)]))
print(f"📏 アヒルの移動量: {duck_movement:.4f}m")
if duck_movement > 0.001:  # 1mm以上動いていれば掴まれている可能性
    print("✅ アヒルが掴まれている可能性があります")
else:
    print("⚠️ アヒルが掴まれていない可能性があります")

# ステップ6: 持ち上げる
print(f"\n⬆️ ステップ6: 持ち上げる")
current_pos = get_arm_position()
lift_height = 0.3  # 30cm上に持ち上げ
lift_target = [current_pos[0], current_pos[1], current_pos[2] + lift_height]

print(f"⬆️ {lift_height*100:.0f}cm上に持ち上げます...")
print(f"🎯 持ち上げ目標: [{lift_target[0]:.4f}, {lift_target[1]:.4f}, {lift_target[2]:.4f}]")

# 段階的に持ち上げる（5段階）
num_lift_steps = 5
for step in range(num_lift_steps):
    current_pos = get_arm_position()
    progress = (step + 1) / num_lift_steps
    intermediate_lift = [
        current_pos[i] + (lift_target[i] - current_pos[i]) * progress
        for i in range(3)
    ]
    print(f"⬆️ 持ち上げ中... {step + 1}/{num_lift_steps} (Z: {intermediate_lift[2]:.3f}m)")
    move_arm_to(intermediate_lift)
    time.sleep(0.3)

# 最終結果を表示
final_pos = get_arm_position()
duck_pos_final, _ = p.getBasePositionAndOrientation(duck_id)

print("\n" + "=" * 60)
print("📊 最終結果")
print("=" * 60)
print(f"🤖 最終的なアームの位置: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
print(f"🦆 最終的なアヒル位置: [{duck_pos_final[0]:.4f}, {duck_pos_final[1]:.4f}, {duck_pos_final[2]:.4f}]")

# アヒルが持ち上げられたか確認
initial_duck_z = duck_pos[2]
final_duck_z = duck_pos_final[2]
lift_amount = final_duck_z - initial_duck_z

print(f"📏 アヒルの持ち上げ量: {lift_amount:.4f}m ({lift_amount*100:.1f}cm)")
if lift_amount > 0.1:
    print("✅ アヒルを正常に持ち上げました！")
else:
    print("⚠️ アヒルが持ち上げられていない可能性があります")

print("=" * 60)

print("\nEnterキーを押すと終了します...")
try:
    input()
except KeyboardInterrupt:
    pass

p.disconnect()
