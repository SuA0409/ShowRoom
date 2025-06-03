import viser
import torch
import numpy as np
import cv2
from pyngrok import ngrok, conf
import socket
import warnings

from KD_Fast3R.fast3r_to_spr import spr, postprocess

# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 빈 포트 탐색
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 운영체제가 자동으로 포트 할당
        return s.getsockname()[1]

data = torch.load('/content/drive/MyDrive/content.pt') # 데모 pt

# list[np.nparray 형태임]
data_preds = [data['preds'][i]['pts3d_in_other_view'].cpu().numpy().squeeze() for i in range(3)]

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

all_points = [] # 모든 xyz 합치기 위함
all_colors = [] # 모든 rgb 합치기 위함
for i in range(num): # 이미지 수 만큼
    image = cv2.imread(f'/content/drive/MyDrive/test_view/{i}.jpg') # 해당 경로에 있는 이미지 로드
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # rgb로 갖고옴
    image = cv2.resize(image, (512, 384)) # 512, 384로 전처리
    image = image.astype(np.float32) / 255.0 # [0,1]로 전처리
    color = np.reshape(image, (-1, 3)) # (N,3)으로 재구성

    all_points.append(pc[i]) # xyz 좌표 추가
    all_colors.append(color) # rgb 좌표 추가

# 모든 포인트 클라우드와 색상 합치기
xyz = np.concatenate(all_points, axis=0) # 원본 xyz 좌표들 합침
rgb = np.concatenate(all_colors, axis=0) # 원본 rgb 값들 합침

# Viser 서버 실행 (단일 서버)
server = viser.ViserServer(host="0.0.0.0", port=port)

# SPR 수행
vertices, colors = spr(
    coords_np_Vx3=xyz,
    colors_np_Vx3=rgb,
    depth=9,
)

# 후처리
vertices, colors = postprocess(xyz, vertices, colors)

# SPR 2번 수행
vertices, colors = spr(
    coords_np_Vx3=vertices,
    colors_np_Vx3=colors,
    depth=9,
)

# 후처리
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
