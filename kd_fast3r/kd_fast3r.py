import torch
import numpy as np
import time
import pymeshlab

from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.models.fast3r import Fast3R

class ShowRoom:
    def __init__(self,
                 model_path,
                 info=True,
                 viz=None
                 ):
        '''Fast3r과 spr 을 활용한 3d reconstruction 클래스
        Args:
            model_path (str): huggingface model 주소
            info (bool): 정보 출력을 컨트롤하는 변수
            viz (ViserMaker): 선언된 ViserMaker 인스턴스
        '''

        # 모델의 처리 device를 선언
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # fast3r pretrained모델을 불러오는 코드
        self.model = Fast3R.from_pretrained(model_path).to(self.device)

        self.info = info
        self.viz = viz

        self.room = None

    # MultiViewDUSt3RLitModule에 camera pose 추정하는 함수 사용 하여 카메라 포즈 추정
    def _get_camera_pose(self, pred):
        poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
            pred,
            niter_PnP=100,
            focal_length_estimation_method='first_view_from_global_head'
        )
        camera_poses = poses_c2w_batch[0]

        # 2d에게 전달해줄 camera pose를 list형태로 저장
        self.pose = [pose.tolist() for pose in camera_poses]

    # Fast3r 모델을 활용한 3d point could 및 camera_pose 추정
    def _predict(self):
        # 전처리 된 이미지를 room과 color 형태로 받음
        images, color = self.room
        sample = len(images)

        for i, image in enumerate(images):
        # input data를 device 타입에 맞게 조정
            images[i]['img'] = image['img'].to(self.device)
            images[i]['true_shape'] = image['true_shape'].to(self.device)

        # model에 대한 정보 출력
        if self.info:
            print(f"    받은 이미지의 개수 : {sample}")
            print(f"    room이 가지고 있는 keys: {list(images[0].keys())}")
            print(f"    이미지의 shape : {images[0]['img'][0].shape}")
            print(f"    데이터 처리 type : {self.device}")

        print(f"    모델 추론 시작 ! ")
        start_time = time.time()

        with torch.no_grad():
            pred = self.model(images)

        # point cloud와 color를 (N, 3) np.ndarray 형태로 변환
        self.point_cloud = np.concatenate([np.reshape(pred[i]['pts3d_in_other_view'].cpu().numpy().squeeze(), (-1, 3)) for i in range(sample)], axis=0)
        self.color = np.concatenate([np.reshape(color[i], (-1, 3)) for i in range(sample)], axis=0)

        if self.info:
            print(f"    모델 추론 완료 ! ({time.time()-start_time:.2f}s)")

        # camera_pose를 추정하여 저장하는 함수, self.pose에 저장
        self._get_camera_pose(pred)

        return self.point_cloud, self.color

    def _post_process(self, point_cloud, vertices, colors):

        """
        뜬금없이 지붕이 생기거나 원반형같이 나오는 등의 이상한 포인트 클라우드를 후처리하는 함수 (입출력: numpy)

        Args:
            point_cloud (np.ndarray): (N, 3) 원본 포인트 클라우드 정점 위치 좌표
            vertices (np.ndarray): (N, 3) 생성된 포인트 클라우드 정점 위치 좌표
            colors (np.ndarray): (N, 3) 생성된 포인트 클라우드 정점 색상 좌표

        Returns:
            vertices (np.ndarray): (N, 3) 후처리된 포인트 클라우드 정점 위치 좌표
            color (np.ndarray): (N, 3) 후처리된 포인트 클라우드 정점 색상 좌표
        """

        # SPR 입력 범위 저장
        x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])  # x 범위 설정
        y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])  # y 범위 설정
        z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])  # z 범위 설정

        # 범위를 벗어난 포인트 제거
        in_bounds_mask = (  # 출력 정점이 입력 포인트의 범위 내에 있는지 확인하는 마스크
                (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
                (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
        )
        vertices = vertices[in_bounds_mask]  # 범위 안의 정점만 살림
        color = colors[in_bounds_mask]  # 범위 안의 색만 살림

        return vertices, color

    def _spr(self, point_cloud, color, depth=9):
        """
        SPR 기반 메시 재구성 함수 (입출력: numpy)

        Args:
            point_cloud (np.ndarray): (N, 3) 원본 포인트 클라우드 정점 좌표
            color (np.ndarray): (N, 3) RGB 색상, [0~1] 범위
            depth (int): Poisson reconstruction 깊이

        Returns:
            vertices (np.ndarray): (N, 3) 메시 정점 좌표
            colors (np.ndarray): (N, 3) 메시 정점 색상 [0~1]
        """

        coords = point_cloud  # point_cloud(N,3)
        colors = color  # color(N,3)

        ms = pymeshlab.MeshSet()
        colors_4 = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)  # N,3 -> N,4

        m = pymeshlab.Mesh(vertex_matrix=coords, v_color_matrix=colors_4)  # 좌표와 색상만으로 메쉬 객체 생성

        ms.add_mesh(m)
        ms.apply_filter('compute_normal_for_point_clouds')  # 법선 벡터 자동 계산
        ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=depth)  # 포아송 재구성 사용

        vertices = ms.current_mesh().vertex_matrix()  # 정점(포인트 클라우드)
        colors = ms.current_mesh().vertex_color_matrix()[:, :3]  # RGBA → RGB

        vertices, color = self._post_process(point_cloud, vertices, colors)  # 후처리

        return vertices, color

    # reconstruction을 하는 main 함수
    def reconstruction(self):
        self._predict()

        self.viz.add_point_cloud('ShowRoom', self.point_cloud, self.color)

    def building_spr(self,
                     depth=9,
                     repeat=2,
                     ):
        '''spr을 실행하고 viser에 표시하는 함수
        Args:
            depth (int): spr의 depth를 정의
            repeat (int): spr의 반복 횟수
        '''

        vertices, color = self.point_cloud, self.color

        for i in range(repeat):
            start_time = time.time()
            vertices, color = self._spr(vertices, color, depth=depth)

            self.viz.add_point_cloud(f'생성된 spr_{i+1}', vertices, color)

            print(f'SPR {i+1}회 적용 완료! ({time.time() - start_time:.2f})')