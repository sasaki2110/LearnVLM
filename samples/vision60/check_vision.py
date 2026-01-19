import pybullet as p
import pybullet_data

p.connect(p.DIRECT) # GUIなしで高速確認
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Vision60のパスを指定（環境に合わせて調整してください）
robot_id = p.loadURDF("quadruped/vision60.urdf")

print(f"\n--- Vision60 Joint List (Total: {p.getNumJoints(robot_id)}) ---")
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    joint_type = info[2] # 0: revolute, 4: fixed
    if joint_type == 0: # 動かせる関節だけ表示
        print(f"Index: {i} | Name: {joint_name}")
p.disconnect()

