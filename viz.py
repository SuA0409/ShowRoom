import viser
from pyngrok import ngrok, conf
import socket
import viser
import numpy as np

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 운영체제가 자동으로 포트 할당
        return s.getsockname()[1]

# 서버 주소 할당
def make_viser_server(token):
    assert isinstance(token, str), 'Token must be a string'

    port = find_free_port() # 빈 포트 추출
    conf.get_default().auth_token = token # Ngrok 설정

    url = ngrok.connect(port, "http") # 새 Ngrok 터널 생성
    server = viser.ViserServer(host="0.0.0.0", port=port) # Viser 서버 실행 (단일 서버)

    print(f'** {url} **')

    return url, server # url, server 반환 (url 고정한 채로 서버는 계속 쓸 수 있음)

class ViserMaker:
    def __init__(self,
                 token,
                 point_size=0.001,
                 data_path='/content/drive/MyDrive/Final_Server/Input/Pts/fast3r_output.npz'
                 ):
        ''' 3d point_cloud를 시각화 하는 viser 기반 클래스
        Args:
            token (str): ngrok에서 부여 받은 토큰
            point_size (float): viser에서 표현할 point당 크기
            data_path (str): point cloud가 저장된 위치
        '''
        self.ngrok_url, self.server = make_viser_server(token)
        self.point_size = point_size

        self._build_viser()

        # .npz 파일의 경로, 본 데이터는 key로 'point_cloud'와 'color' 보유
        self.data_path = data_path

    # viser를 초기화 하는 함수
    def _build_viser(self):

        @self.server.on_client_connect
        def on_client_connect(client: viser.ClientHandle) -> None:
            with client.atomic():
                client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
                client.camera.look_at = (-0.00352821, -0.01143425, 0.0154939)

            client.flush()

        self.server.scene.set_up_direction((0.0, -1.0, 0.0))
        self.server.scene.world_axes.visible = False

    # point_cloud를 viser에 추가하는 함수
    def add_point_cloud(self, name, point_cloud=None, color=None):
        if point_cloud is None and color is None:
            data = np.load(self.data_path, allow_pickle=True)
            point_cloud, color = data['point_cloud'], data['color']

        self.server.add_point_cloud(
            name=name,
            points=point_cloud,
            colors=color,
            point_size=self.point_size
        )