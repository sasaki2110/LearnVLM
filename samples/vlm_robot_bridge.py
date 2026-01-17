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
arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
# アヒルをランダムな位置に置く（今回はテスト用に固定）
duck_target_pos = [0.6, 0.3, 0.0] 
duck_id = p.loadURDF("duck_vhacd.urdf", basePosition=duck_target_pos)

# シミュレーションを少し進める（安定化のため）
for _ in range(500):
    p.stepSimulation()
    time.sleep(1./240.)

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
    # 前回と同じカメラ設定
    view_matrix = p.computeViewMatrix([1.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.33, 0.1, 100.0)
    _, _, rgb_img, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix)
    return Image.fromarray(np.reshape(rgb_img, (480, 640, 4))[:, :, :3].astype('uint8'))

# --- 3. 座標変換の魔法 (2D -> 3D) - 調整版 ---
def map_vlm_to_world(vlm_coords, camera_eye=[1.0, 0.0, 1.0], camera_target=[0.5, 0.0, 0.0], 
                     fov=60, image_size=(640, 480), plane_z=0.0, **kwargs):
    """
    簡易マッピング（元の考え方を維持しつつ、パラメータを調整）
    
    Args:
        vlm_coords: [ymin, xmin, ymax, xmax] の正規化座標 (0~1)
        camera_eye: カメラの位置 [x, y, z]
        camera_target: カメラが向く先 [x, y, z]
        fov: 視野角（度）
        image_size: (width, height)
        plane_z: 物体が置かれている平面のZ座標
    """
    # vlm_coords = [ymin, xmin, ymax, xmax] 
    y_center = (vlm_coords[0] + vlm_coords[2]) / 2
    x_center = (vlm_coords[1] + vlm_coords[3]) / 2
    
    # バウンディングボックスのサイズから奥行きを推定
    bbox_width = vlm_coords[3] - vlm_coords[1]  # x方向のサイズ
    bbox_height = vlm_coords[2] - vlm_coords[0]  # y方向のサイズ
    bbox_size = max(bbox_width, bbox_height)  # 大きい方を使用
    
    # 元の簡易マッピングの考え方を維持
    # 画像のy座標（上下）→ ワールドのX座標（奥行き）
    # 画像のx座標（左右）→ ワールドのY座標（左右）
    # 
    # 実際のデータから逆算してパラメータを調整:
    # 実際のアヒル位置 [0.6073, 0.2827]、VLM座標 y_center=0.63, x_center=0.54
    # 
    # 元の簡易マッピング: 
    #   world_x = 0.8 - (y_center * 0.6) → 範囲 [0.2, 0.8]
    #   world_y = 0.4 - (x_center * 0.8) → 範囲 [-0.4, 0.4]
    # 
    # 実際のデータに合わせて微調整:
    #   0.6073 ≈ 0.8 - (0.63 * 0.306) → depth_coeff ≈ 0.306
    #   0.2827 ≈ 0.4 - (0.54 * 0.217) → width_coeff ≈ 0.217
    # 
    # ただし、1つのデータポイントだけでは不十分なので、
    # 距離ファクターを受け取る（デフォルトは1.0）
    distance_factor = kwargs.get('distance_factor', 1.0)
    
    # 元のパラメータをベースに、より広い範囲をカバーするように調整
    # 距離に応じてスケールを調整
    world_x = 0.8 - (y_center * 0.4 * distance_factor)  # 範囲 [0.4, 0.8] に調整（より広く）
    world_y = 0.4 - (x_center * 0.5 * distance_factor)  # 範囲 [-0.1, 0.4] に調整（より広く）
    
    # Z座標は平面の高さ + アヒルの実際の高さ
    # バウンディングボックスサイズによる補正は小さく
    estimated_z = plane_z + 0.016 + (1.0 - bbox_size) * 0.02  # 補正項を小さく
    
    result = [float(world_x), float(world_y), float(estimated_z)]
    
    return result

# --- 4. 視覚フィードバックループ（撮影→移動→撮影→移動） ---
def move_arm_to(target_pos):
    """アームを目標位置に移動"""
    joint_poses = p.calculateInverseKinematics(arm_id, 6, target_pos)
    for i in range(len(joint_poses)):
        p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_poses[i])
    for _ in range(200): 
        p.stepSimulation()
        time.sleep(1./240.)

def get_arm_position():
    """現在のアーム先端位置を取得"""
    arm_ee_state = p.getLinkState(arm_id, 6)
    return arm_ee_state[0]

def detect_duck_position(show_debug=False, current_arm_pos=None):
    """
    VLMを使ってアヒルの位置を検出
    アームが近づくと、バウンディングボックスが大きくなり、より正確な位置推定が可能
    """
    img = capture_image()
    enc_image = model.encode_image(img)
    coords = model.answer_question(enc_image, "Point out the duck with a bounding box.", tokenizer)
    
    try:
        num_coords = eval(coords)
        bbox_size = max(num_coords[3] - num_coords[1], num_coords[2] - num_coords[0])
        
        # バウンディングボックスのサイズから信頼度を計算
        # 大きい = 近い = より正確
        # ただし、アームが近づきすぎると、バウンディングボックスが大きくなりすぎる可能性もある
        confidence = min(bbox_size * 2.0, 1.0)  # 0~1の範囲に正規化
        
        # アームが近い場合、マッピングのパラメータを調整
        # アームが近づくと、視野角の影響が小さくなり、より正確な位置推定が可能
        if current_arm_pos is not None:
            # アームとカメラターゲットの距離を計算
            arm_to_target = np.linalg.norm(np.array(current_arm_pos[:2]) - np.array([0.5, 0.0]))
            # 距離が近いほど、マッピングのスケールを調整
            # 近い場合、より細かい調整が可能
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
            print(f"🔍 マッピング詳細: y_center={y_center:.3f}, x_center={x_center:.3f}, "
                  f"bbox_size={bbox_size:.3f}, confidence={confidence:.3f}, "
                  f"distance_factor={distance_factor:.3f}, world=[{target_3d[0]:.3f}, {target_3d[1]:.3f}, {target_3d[2]:.3f}]")
        
        # バウンディングボックスのサイズから距離を推定
        # 大きい = 近い、小さい = 遠い
        # 経験的な式: 距離 ≈ 1 / (bbox_size * scale_factor)
        # bbox_sizeが0.1のとき距離約0.5m、0.5のとき距離約0.1mと仮定
        estimated_distance = 0.5 / (bbox_size + 0.01)  # 0除算を避けるため+0.01
        estimated_distance = max(0.05, min(1.0, estimated_distance))  # 0.05m~1.0mの範囲に制限
        
        return target_3d, num_coords, confidence, bbox_size, estimated_distance
    except:
        print(f"❌ 座標がうまく取れませんでした: {coords}")
        return None, None, 0.0, 0.0, 1.0

# --- 5. 段階的な接近アプローチ ---
print("=" * 60)
print("🎯 段階的な接近アプローチ開始")
print("=" * 60)

# ステップ1: VLMで目標位置を決定
print("\n📸 ステップ1: VLMで目標位置を決定")
current_pos = get_arm_position()
result = detect_duck_position(show_debug=True, current_arm_pos=current_pos)
if result[0] is None:
    print("❌ 目標位置の検出に失敗しました")
    exit(1)

target_3d, num_coords, confidence, _, _ = result
print(f"🎯 目標位置: [{target_3d[0]:.4f}, {target_3d[1]:.4f}, {target_3d[2]:.4f}]")

# 目標位置の10cm上を計算
approach_height = 0.10  # 10cm
target_above = [target_3d[0], target_3d[1], target_3d[2] + approach_height]
print(f"📍 接近位置（目標の{approach_height*100:.0f}cm上）: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")

# ステップ2: 目標位置の10cm上まで段階的に移動（10段階）
print(f"\n📈 ステップ2: 目標位置の{approach_height*100:.0f}cm上まで段階的に移動（10段階）")
num_steps_approach = 10

for step in range(num_steps_approach):
    print(f"\n--- 段階 {step + 1}/{num_steps_approach} ---")
    
    # 現在位置を取得
    current_pos = get_arm_position()
    
    # 各段階でVLMで位置取得をやり直す
    result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
    if result[0] is None:
        print("⚠️ 検出失敗、前回の目標位置を使用")
    else:
        new_target, _, _, _, _ = result
        # 目標位置を更新（X, Yのみ、Zは10cm上を維持）
        target_above[0] = new_target[0]
        target_above[1] = new_target[1]
        print(f"🔄 目標位置を更新: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
    
    # 現在位置から目標位置への進捗を計算
    progress = (step + 1) / num_steps_approach
    intermediate_pos = [
        current_pos[i] + (target_above[i] - current_pos[i]) * progress
        for i in range(3)
    ]
    
    print(f"📍 現在位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    print(f"🎯 中間目標: [{intermediate_pos[0]:.4f}, {intermediate_pos[1]:.4f}, {intermediate_pos[2]:.4f}]")
    
    # 移動
    move_arm_to(intermediate_pos)
    time.sleep(0.3)

# ステップ3: 10cm上に到達したら、真上に移動するまでループ
print(f"\n🎯 ステップ3: 真上に移動するまでループ")
max_align_iterations = 10
align_threshold = 0.02  # 2cm以内なら真上とみなす

for align_iter in range(max_align_iterations):
    print(f"\n--- 位置合わせ {align_iter + 1}/{max_align_iterations} ---")
    
    current_pos = get_arm_position()
    
    # VLMで位置取得
    result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
    if result[0] is None:
        print("❌ 検出失敗、終了します")
        break
    
    new_target, _, _, _, _ = result
    
    # X, Y方向の誤差を計算
    xy_error = np.sqrt((current_pos[0] - new_target[0])**2 + (current_pos[1] - new_target[1])**2)
    print(f"📏 X, Y方向の誤差: {xy_error:.4f}m")
    
    if xy_error < align_threshold:
        print(f"✅ 真上に到達しました！（誤差: {xy_error:.4f}m < {align_threshold:.3f}m）")
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
    
    print(f"🎯 目標位置: [{target_above[0]:.4f}, {target_above[1]:.4f}, {target_above[2]:.4f}]")
    move_arm_to(intermediate_pos)
    time.sleep(0.3)

# ステップ4: 真上に来たら、少しずつ下がる（10段階）
print(f"\n📉 ステップ4: 少しずつ下がる（バウンディングボックスから距離を推定）")
num_steps_descend = 10
final_target = target_3d.copy()  # 最終的な目標位置

current_pos = get_arm_position()
start_z = current_pos[2]

# バウンディングボックスのサイズから適切な停止位置を計算
# 最初に距離を推定
result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
if result[0] is not None:
    _, _, _, bbox_size, estimated_distance = result
    # 推定距離から安全マージンを計算
    # 距離が近いほど、より小さなマージンで停止
    safety_margin = max(0.03, min(0.08, estimated_distance * 0.15))  # 3cm~8cmの範囲
    end_z = final_target[2] + safety_margin
    print(f"📏 推定距離: {estimated_distance:.3f}m, バウンディングボックスサイズ: {bbox_size:.3f}")
    print(f"🛡️ 安全マージン: {safety_margin*100:.1f}cm, 停止位置Z: {end_z:.4f}m")
else:
    # 検出失敗時はデフォルトの安全マージン（5cm）を使用
    safety_margin = 0.05
    end_z = final_target[2] + safety_margin
    print(f"⚠️ 検出失敗、デフォルトの安全マージン（5cm）を使用")

for step in range(num_steps_descend):
    print(f"\n--- 下降段階 {step + 1}/{num_steps_descend} ---")
    
    current_pos = get_arm_position()
    
    # 各段階でVLMで位置取得をやり直す
    result = detect_duck_position(show_debug=False, current_arm_pos=current_pos)
    if result[0] is None:
        print("⚠️ 検出失敗、前回の目標位置を使用")
    else:
        new_target, _, _, bbox_size, estimated_distance = result
        # X, Yを更新
        final_target[0] = new_target[0]
        final_target[1] = new_target[1]
        
        # バウンディングボックスのサイズから距離を再推定し、安全マージンを更新
        # アームが近づくと、バウンディングボックスが大きくなり、距離が短くなる
        new_safety_margin = max(0.03, min(0.08, estimated_distance * 0.15))
        new_end_z = final_target[2] + new_safety_margin
        
        # より安全な位置（高い位置）を選択
        if new_end_z > end_z:
            end_z = new_end_z
            safety_margin = new_safety_margin
        
        print(f"🔄 目標位置を更新: [{final_target[0]:.4f}, {final_target[1]:.4f}, {final_target[2]:.4f}]")
        print(f"📏 推定距離: {estimated_distance:.3f}m, バウンディングボックス: {bbox_size:.3f}, "
              f"安全マージン: {safety_margin*100:.1f}cm")
    
    # Z方向の進捗を計算
    progress = (step + 1) / num_steps_descend
    target_z = start_z - (start_z - end_z) * progress
    
    # 中間位置を計算
    intermediate_pos = [final_target[0], final_target[1], target_z]
    
    print(f"📍 現在位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    print(f"🎯 中間目標: [{intermediate_pos[0]:.4f}, {intermediate_pos[1]:.4f}, {intermediate_pos[2]:.4f}]")
    
    # 接触を避けるため、目標位置より下に下がりすぎないようにチェック
    if target_z < end_z:
        target_z = end_z
        intermediate_pos[2] = end_z
        print(f"⚠️ 安全マージンを維持: Z={end_z:.4f}m")
    
    # 移動
    move_arm_to(intermediate_pos)
    time.sleep(0.3)

# ステップ5: 触れる位置に来たら掴む（グリッパーがないので、位置まで移動するだけ）
print(f"\n🤏 ステップ5: 最終位置に到達")
final_pos = get_arm_position()
print(f"📍 最終位置: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
print("💡 注: このロボットアームにはグリッパーがありません。位置まで移動しました。")

# 最終結果を表示
final_pos = get_arm_position()
print("\n" + "=" * 60)
print("📊 最終結果")
print("=" * 60)
print(f"🤖 最終的なアームの位置: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")

# 実際のアヒル位置との比較
duck_pos, _ = p.getBasePositionAndOrientation(duck_id)
print(f"🦆 実際のアヒル位置: [{duck_pos[0]:.4f}, {duck_pos[1]:.4f}, {duck_pos[2]:.4f}]")

final_diff = [final_pos[i] - duck_pos[i] for i in range(3)]
final_distance = np.sqrt(sum([d**2 for d in final_diff]))
print(f"📏 最終誤差: {final_distance:.4f}m")

# X, Y方向の誤差も表示
xy_error = np.sqrt(final_diff[0]**2 + final_diff[1]**2)
z_error = abs(final_diff[2])
print(f"📏 X, Y方向の誤差: {xy_error:.4f}m")
print(f"📏 Z方向の誤差: {z_error:.4f}m")
print("=" * 60)

print("\nEnterキーを押すと終了します...")

try:
    input()
except KeyboardInterrupt:
    pass

p.disconnect()