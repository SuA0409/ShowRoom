import viser
import torch
import numpy as np
import cv2
from pyngrok import ngrok, conf
import socket
import warnings
import glob
import os

from fast3r_to_spr import spr

# 빈 포트 탐색
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 운영체제가 자동으로 포트 할당
        return s.getsockname()[1]

# 서버 주소 할당
def make_server(token):
    port = find_free_port() # 빈 포트 추출
    conf.get_default().auth_token = token # Ngrok 설정
    
    url = ngrok.connect(port, "http") # 새 Ngrok 터널 생성
    server = viser.ViserServer(host="0.0.0.0", port=port) # Viser 서버 실행 (단일 서버)
    return url, server # url, server 반환 (url 고정한 채로 서버는 계속 쓸 수 있음)

# viser로 원본 3d와 spr*2로 생성 3d를 보여주는 함수
def viz(pc, server, path, size=(512, 384)):
    # 딕셔너리 numpy를 list numpy로 바꿈
    pc = [pc[k] for k in pc]
    # 해당 경로의 jpg 파일들
    all_image = sorted(glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True))
    # 이미지 갯수
    num = len(all_image)
    all_points = [] # 모든 xyz 합치기 위함
    all_colors = [] # 모든 rgb 합치기 위함
    for i, p in enumerate(pc): # 이미지 수 만큼
        image = all_image[i] # i번째 이미지
        image = cv2.imread(image) # 읽음
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # rgb로 갖고옴
        image = cv2.resize(image, size) # 512, 384로 전처리(default)
        image = image.astype(np.float32) / 255.0 # [0,1]로 전처리
        color = np.reshape(image, (-1, 3)) # (N,3)으로 재구성

        all_points.append(p) # xyz 좌표 추가
        all_colors.append(color) # rgb 좌표 추가
    
    # 모든 포인트 클라우드와 색상 합치기
    xyz = np.concatenate(all_points, axis=0) # 원본 xyz 값들 합침
    rgb = np.concatenate(all_colors, axis=0) # 원본 rgb 값들 합침
    
    # 먼저 합쳐진 원본 포인트 클라우드 시각화
    server.scene.add_point_cloud(
        name="원본 포인트 클라우드",
        points=xyz,
        colors=rgb,
        point_size=0.001
    )
    
    # SPR 수행
    vertices, colors = spr(xyz, rgb)

    server.scene.add_point_cloud(
        name="생성 포인트 클라우드1",
        points=vertices,
        colors=colors,
        point_size=0.001
    )
    
    # SPR 2번 수행
    vertices, colors = spr(vertices, colors)
    
    # SPR 결과 시각화
    server.scene.add_point_cloud(
        name="생성 포인트 클라우드2",
        points=vertices,
        colors=colors,
        point_size=0.001
    )

    print("✅ 접속 후 아래 셀에서 Enter를 누르면 서버가 종료됩니다.")
    input("Press Enter to stop viser...")
    ngrok.kill()  # Ngrok 터널 종료

def more_spr(vertices, colors, depth=9, server):
    '''
    depth도 조종 가능
    '''

    vertices, colors = spr(vertices, colors, depth)
    
    # SPR 결과 시각화
    server.scene.add_point_cloud(
        name="생성 포인트 클라우드",
        points=vertices,
        colors=colors,
        point_size=0.001
    )

    return vertices, colors

'''

# 사용 예:
from kd_fast3r.utils import make_server, viz

url, server = make_server('your token') # url과 server 받아옴
pc = np.load(pc_ndarray) # pc 인풋 가져옴(fast3r 출력값)
viz(pc, server, path='/content/drive/MyDrive/test_view') # 3d 시각화

# 만약에 spr 요청시
vertices, colors = more_spr(vertices, colors)

'''
