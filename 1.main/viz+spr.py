import viser
import torch
import numpy as np
import cv2
from pyngrok import ngrok, conf
import socket
import warnings
import os

from KD_Fast3R.fast3r_to_spr import spr, postprocess

# 빈 포트 탐색
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 운영체제가 자동으로 포트 할당
        return s.getsockname()[1]

# 서버 주소 할당
def make_server():
    # 빈 포트 추출
    port = find_free_port()
    
    # Ngrok 설정
    conf.get_default().auth_token = "2xwkthyPz15CsSbartjgnt9aQde_3RoEvuB7Mz7oHHzuDJFia"
    
    # 새 Ngrok 터널 생성
    url = ngrok.connect(port, "http")
    print(f"🔗 Viser 접속 링크: {url}")

    # Viser 서버 실행 (단일 서버)
    server = viser.ViserServer(host="0.0.0.0", port=port)

    return url, server # url, server = make_server() 이렇게 하면 될듯??

def viz(pc, server, path, size=(512, 384)):
    # 경고 무시
    # warnings.filterwarnings("ignore", category=UserWarning)
    
    # data = torch.load(pt) # 데모 pt
    
    # # list[np.nparray 형태임]
    # data_preds = [data['preds'][i]['pts3d_in_other_view'].cpu().numpy().squeeze() for i in range(3)]

    # # 데이터 로드
    # a = torch.load('/content/drive/MyDrive/content.pt', weights_only=True)
    # num = len(a['preds'])  ==> num
    
    # # 포인트 클라우드 좌표
    # pc = [np.reshape(a['preds'][i]['pts3d_in_other_view'].cpu().numpy().squeeze(), (-1, 3)) for i in range(num)]
    # pc = np.round(pc, 5)  # ==> pc
    
    all_points = [] # 모든 xyz 합치기 위함
    all_colors = [] # 모든 rgb 합치기 위함
    for i, p in enumerate(pc): # 이미지 수 만큼
        image = os.path.join(path, f'{i}.jpg') # 해당 경로에 있는 이미지 로드
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # rgb로 갖고옴
        image = cv2.resize(image, (512, 384)) # 512, 384로 전처리
        image = image.astype(np.float32) / 255.0 # [0,1]로 전처리
        color = np.reshape(image, (-1, 3)) # (N,3)으로 재구성

        all_points.append(p) # xyz 좌표 추가
        all_colors.append(color) # rgb 좌표 추가
    
    # 모든 포인트 클라우드와 색상 합치기
    xyz = np.concatenate(all_points, axis=0) # 원본 xyz 값들 합침
    rgb = np.concatenate(all_colors, axis=0) # 원본 rgb 값들 합침
    
    # 합쳐진 포인트 클라우드 시각화
    server.scene.add_point_cloud(
        name="원본 포인트 클라우드",
        points=xyz,
        colors=rgb,
        point_size=0.001
    )
    
    # SPR 수행
    vertices, colors = spr(xyz, xyz, rgb)
    
    # SPR 2번 수행
    vertices, colors = spr(xyz, vertices, colors)
    
    # SPR 결과 시각화
    server.scene.add_point_cloud(
        name="생성 포인트 클라우드",
        points=vertices,
        colors=colors,
        point_size=0.001
    )

    print("✅ 접속 후 아래 셀에서 Enter를 누르면 서버가 종료됩니다.")
    input("Press Enter to stop viser...")
    ngrok.kill()  # Ngrok 터널 종료

# 사용 예:
# from 1.main import make_server, viz

# url, server = make_server() # url과 server 받아옴
# pc = np.load('/content/drive/MyDrive/views.npz') # pc 인풋 가져옴 !! 넘피로 어차피 나오니 실사용엔 필요 없을 듯
# pc = [pc[k] for k in pc] # dict를 list로 바꿈
# viz(pc, server, path='/content/drive/MyDrive/test_view') # 3d 시각화
