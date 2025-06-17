import torch
import numpy as np
import time

from kd_fast3r.utils.fast3r_to_spr import spr
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.models.fast3r import Fast3R


class ShowRoom:
    def __init__(self,
                 model_path='jedyang97/Fast3R_ViT_Large_512',
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
        self.model.eval()

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
        self.point_cloud = [np.reshape(pred[i]['pts3d_in_other_view'].cpu().numpy().squeeze(), (-1, 3)) for i in
                            range(sample)]
        self.color = [np.reshape(color[i], (-1, 3)) for i in range(sample)]

        if self.info:
            print(f"    모델 추론 완료 ! ({time.time() - start_time:.2f}s)")

        # camera_pose를 추정하여 저장하는 함수, self.pose에 저장
        self._get_camera_pose(pred)

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

        vertices = np.concatenate(self.point_cloud, axis=0)
        color = np.concatenate(self.color, axis=0)

        for i in range(repeat):
            start_time = time.time()
            vertices, color = spr(vertices, color, depth=depth)

            self.viz.add_point_cloud(f'spr {i + 1}', vertices, color, False)

            print(f'SPR {i + 1}회 적용 완료! ({time.time() - start_time:.2f})')