from pyngrok import ngrok, conf
import socket
import viser
import numpy as np

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 운영체제가 자동으로 포트 할당
        return s.getsockname()[1]

class ViserMaker:
    def __init__(self,
                 token=None,
                 point_size=0.001,
                 ):
        ''' 3d point_cloud를 시각화 하는 viser 기반 클래스
        Args:
            token (str): ngrok에서 부여 받은 토큰
            point_size (float): viser에서 표현할 point당 크기
        '''
        self.point_size = point_size

        self.make_viser_server(token)
        self._build_viser()

        self.is_gen = "False"
        self.move = 0
        self.server_list = list()

    # 서버 주소 할당
    def make_viser_server(self, token):
        try:
            assert isinstance(token, str)

            port = find_free_port()  # 빈 포트 추출
            conf.get_default().auth_token = token  # Ngrok 설정

            self.ngrok_url = ngrok.connect(port, "http")  # 새 Ngrok 터널 생성
            self.server = viser.ViserServer(host="0.0.0.0", port=port)  # Viser 서버 실행 (단일 서버)

            print(f'*** {self.ngrok_url} ***')
        except:
            print('Start in Local...')
            self.server = viser.ViserServer()  # Viser 서버 실행 (로컬)


    # viser를 초기화 하는 함수
    def _build_viser(self):

        # viser에서의 시점을 초기화 하는 코드
        @self.server.on_client_connect
        def on_client_connect(client: viser.ClientHandle) -> None:
            with client.atomic():
                client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
                client.camera.look_at = (-0.00352821, -0.01143425, 0.0154939)

            client.flush()

        self.server.scene.set_up_direction((0.0, -1.0, 0.0))
        self.server.scene.world_axes.visible = False

    def _clear(self):
        '''viser에 등록된 point cloud와 color를 지우는 함수'''
        for sl in reversed(self.server_list):
            sl.remove()
        self.server_list = list()

    def add_point_cloud(self,
                        name='default',
                        point_clouds=None,
                        colors=None,
                        init_viz=True):
        '''point_cloud를 viser에 추가하는 함수
        Args:
            name (str): viser에서 구분할 point cloud의 이름
            point_clouds (Union[np.ndarray, list]): 화면에 표시할 point cloud
            colors (Union[np.ndarray, list]): 화면에 표시할 point cloud와 매칭되는 color
            init_viz (bool): viser에 넣은 point could 초기 on/off 설정
        '''
        # srp과 fast3r 결과를 다형성으로 구분하여 실행
        if isinstance(point_clouds, list):
            # generation만 viser에 따로 업데이트
            if self.is_gen == 'True':
                self.move -= point_clouds[0][0]
                point_cloud =  np.stack(point_clouds)[-1] - self.move
                color = np.stack(colors)[-1]
                name = "Generation"
                print(name)
                init_viz = False
            else:
                self._clear()
                point_cloud = np.concatenate(point_clouds, axis=0)
                color = np.concatenate(colors, axis=0)
                self.move = point_cloud[0, :]
        else:
            point_cloud = point_clouds
            color = colors
        self.server_list.append(self.server.add_point_cloud(
            name=name,
            points=point_cloud*2,
            colors=color,
            point_size=self.point_size*2,
            visible=init_viz
        ))