import viser
import torch
import numpy as np
import cv2
from pyngrok import ngrok
from pyngrok import conf
import socket
import torch
import os
import time
import re
import json
import copy

from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from typing_extensions import LiteralString
from fast3r_to_spr import spr, postprocess

data = torch.load('/content/drive/MyDrive/content.pt') #넣으면 열림

# list[np.nparray 형태임]
data_preds = [data['preds'][i]['pts3d_in_other_view'].cpu().numpy().squeeze() for i in range(3)]
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 운영체제가 자동으로 포트 할당
        return s.getsockname()[1]

# 빈 포트 탐색
port = find_free_port()

# Ngrok 설정
conf.get_default().auth_token = "2xtr40SYpZezDa87vL5L1N2bRmA_4cZfyszzUp23QxB7WeVr1"

# 새 Ngrok 터널 생성
public_url = ngrok.connect(port, "http")
print(f"🔗 Viser 접속 링크: {public_url}")

# 데이터 로드
a = torch.load('/content/drive/MyDrive/content.pt', weights_only=True)
num = len(a['preds'])

# 포인트 클라우드 좌표
pc = [np.reshape(a['preds'][i]['pts3d_in_other_view'].cpu().numpy().squeeze(), (-1, 3)) for i in range(num)]
pc = np.round(pc, 5)

all_points = []
all_colors = []
for i in range(num):
    image = cv2.imread(f'/content/drive/MyDrive/test_view/{i}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 384))
    image = image.astype(np.float32) / 255.0
    color = np.reshape(image, (-1, 3))

    all_points.append(pc[i])
    all_colors.append(color)

# 모든 포인트 클라우드와 색상 합치기
xyz = np.concatenate(all_points, axis=0)
rgb = np.concatenate(all_colors, axis=0)

# Viser 서버 실행 (단일 서버)
server = viser.ViserServer(host="0.0.0.0", port=port)

# SPR 수행 (한 번만 호출)
vertices, colors = spr(
    coords_np_Vx3=xyz,
    colors_np_Vx3=rgb,
    depth=8,
)

vertices, colors = postprocess(xyz, vertices, colors)

vertices, colors = spr(
    coords_np_Vx3=vertices,
    colors_np_Vx3=colors,
    depth=8,
)

vertices, colors = postprocess(xyz, vertices, colors)

# 합쳐진 포인트 클라우드 시각화
server.scene.add_point_cloud(
    name="원본 포인트 클라우드",
    points=xyz,
    colors=rgb,
    point_size=0.001
)

# SPR 결과 시각화
server.scene.add_point_cloud(
    name="생성 포인트 클라우드",
    points=vertices,
    colors=colors,
    point_size=0.001
)

print(f"Combined points shape: {xyz.shape}")
print(f"Combined colors shape: {rgb.shape}")
print(f"Reconstructed vertices shape: {vertices.shape}")
print(f"Reconstructed colors shape: {colors.shape}")

print("✅ 접속 후 아래 셀에서 Enter를 누르면 서버가 종료됩니다.")
input("Press Enter to stop viser...")
ngrok.kill()  # Ngrok 터널 종료
