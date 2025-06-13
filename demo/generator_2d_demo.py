import os
import cv2
import numpy as np
from generate2d.discriminator.discriminator2d import dis_main
from generate2d.generator.stable_diffusion import gen_main

# ====== 설정 ======
image_dir = "/content/ShowRoom/demo/data"
pose_path = "/content/ShowRoom/demo/data/poses.txt"

# ====== 이미지 불러오기 ======
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"이미지 경로가 존재하지 않습니다: {image_dir}")

image_files = sorted([
    f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))
])

request_data = {}
for idx, fname in enumerate(image_files):
    path = os.path.join(image_dir, fname)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"이미지 파일을 읽을 수 없습니다: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    request_data[str(idx)] = img_rgb

# ====== 포즈 파일 불러오기 ======
if not os.path.exists(pose_path):
    raise FileNotFoundError(f"포즈 파일 경로가 존재하지 않습니다: {pose_path}")

with open(pose_path, "r") as f:
    pose_text = f.read()

try:
    safe_env = {"array": np.array, "float32": np.float32}
    pose_list = eval(pose_text, safe_env)
    pose = {"pose": pose_list}
except Exception as e:
    raise ValueError(f"포즈 파일 파싱 실패: {e}")

# ====== Discriminator 실행 ======
result = dis_main(request_data, pose)

# 결과 출력
for item in result:
    print(f"key: {item['key']}, image shape: {item['image'].shape if item['image'] is not None else 'None'}")

# ====== Stable Diffusion Generator 실행 ======
gen_main(result)
