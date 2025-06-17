import os
import cv2
import time
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict

# TensorFlow Keras ConvNeXtTiny (TF 2.11 이상)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input

# spatial_transformer.py 에 정의된 ProjectiveTransformer 클래스
from generate2d.discriminator.spatial_transformer import ProjectiveTransformer

@dataclass
class ProcessorConfig:
    """ShowRoomProcessor 설정을 위한 데이터 클래스"""

    # 가중치 파일 경로
    weight_path: str = 'generate2d/discriminator/weight/Weight_ST_RoomNet_ConvNext.h5'
    ref_img_path: str = 'generate2d/discriminator/ref_img2.png'

    # 모델 설정
    image_size: Tuple[int, int] = (400, 400)
    input_channels: int = 3
    theta_dim: int = 8

    # 정면 판별 설정
    front_view_class_id: int = 1
    center_threshold: float = 0.25
    pixel_threshold: int = 5000

    # GPU 설정 (True: GPU 사용, False: CPU 사용)
    use_gpu: bool = True

class ShowRoomProcessor:
    """ShowRoom 레이아웃 처리를 위한 메인 클래스"""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        ShowRoomProcessor 초기화

        Args:
            config (Optional[ProcessorConfig]): 사용자 정의 설정. 없을 경우 기본 설정 사용
        """

        self.config = config if config is not None else ProcessorConfig()
        self.model = None
        self.theta_model = None

        self._setup_gpu()
        self._build_model()

    def _setup_gpu(self) -> None:
        """GPU 또는 CPU 사용 설정"""

        if not self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("CPU")
        else:
            print("GPU")

    def _build_model(self) -> None:
        """ConvNeXt 기반 모델 구축 및 가중치(.h5) 로드"""

        # 기준 이미지(ref_img2.png) 불러오기
        if not os.path.exists(self.config.ref_img_path):
            raise FileNotFoundError(f"{self.config.ref_img_path} 파일을 찾을 수 없습니다.")

        ref_img = tf.io.read_file(self.config.ref_img_path)
        ref_img = tf.io.decode_png(ref_img, channels=3)  # PNG를 RGB로 디코딩
        ref_img = tf.cast(ref_img, tf.float32) / 51.0  # 0~1 사이 정규화 (원래 코드 비율 유지)
        ref_img = tf.image.resize(ref_img, self.config.image_size)  # 크기 보정
        ref_img = ref_img[tf.newaxis, ...]  # (1, 400, 400, 3) 배치 차원 추가

        # ConvNeXtTiny Base 모델 (include_top=False, pooling='avg')
        base_model = ConvNeXtTiny(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.config.image_size, self.config.input_channels),
            pooling='avg'
        )

        # Theta 값을 예측할 Dense 레이어 추가
        theta_layer = Dense(self.config.theta_dim, name='theta_layer')(base_model.output)

        # ProjectiveTransformer로 Warping 수행
        transformer = ProjectiveTransformer(self.config.image_size)

        # stl: Spatial Transformer Layer의 출력 (정규화되지 않은 형태)
        # 입력 이미지는 ref_img(고정) → theta 값은 trainable
        stl = transformer.transform(ref_img, theta_layer)

        # 최종 모델: 입력 → Theta → Spatial Transformer 변환 출력
        self.model = Model(inputs=base_model.input, outputs=stl)

        # 사전에 학습된 가중치(.h5) 불러오기
        if not os.path.exists(self.config.weight_path):
            raise FileNotFoundError(f"{self.config.weight_path} 파일을 찾을 수 없습니다.")

        self.model.load_weights(self.config.weight_path)
        print("메인 모델 가중치 로드 완료")

        # Theta만 별도로 뽑아내기 위한 서브 모델
        self.theta_model = Model(inputs=base_model.input, outputs=theta_layer)

    def _load_data(self, images, poses):
        """
        서버 요청 데이터를 로드하고 처리

        Args:
            images: 이미지 데이터 (bytes 객체)
            poses (dict): 포즈 데이터 {'pose': [[...], [...], [...]]} 형식

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: 처리된 이미지 리스트와 정렬된 포즈 리스트

        Raises:
            ValueError: 이미지 디코딩 실패 시 발생
        """

        processed_images = []
        processed_poses = []

        try:
            #  Case 1: 서버 형식 - file-like (readable) 이미지 + dict 포즈
            if all(hasattr(f, "read") for f in images.values()) and isinstance(poses, dict) and "pose" in poses:
                for idx, f in enumerate(images.values()):
                    image_bytes = f.read()
                    file_bytes = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"이미지 {idx} 디코딩 실패")
                    processed_images.append(img)

                for pose in poses["pose"]:
                    processed_poses.append(np.array(pose, dtype=np.float32))

            #  Case 2: 데모 형식 - np.ndarray 이미지 + dict 포즈
            elif all(isinstance(f, np.ndarray) for f in images.values()) and isinstance(poses, dict) and "pose" in poses:
                for idx in sorted(images.keys(), key=lambda x: int(x)):
                    img = images[idx]
                    if img is None:
                        raise ValueError(f"이미지 {idx}가 None입니다.")
                    processed_images.append(img)

                for pose in poses["pose"]:
                    processed_poses.append(np.array(pose, dtype=np.float32))

            else:
                raise ValueError(" 지원하지 않는 입력 형식입니다. 이미지(dict)와 포즈(dict['pose'])가 올바른 구조인지 확인하세요.")

            #  포즈 정렬 공통 처리
            processed_poses = self.apply_fast3r_camera_alignment(processed_poses)

            print(f" 데이터 로드 완료: 이미지 {len(processed_images)}개, 포즈 {len(processed_poses)}개")
            return processed_images, processed_poses

        except Exception as e:
            print(f" 데이터 로드 중 오류 발생: {e}")
            return None, None

    def apply_fast3r_camera_alignment(self, pose_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Fast3R에서 사용한 viser 시각화 기준 (카메라 방향 및 상방)으로 포즈 회전 정렬

        Args:
            pose_list (List[np.ndarray]): 4x4 포즈 행렬 리스트

        Returns:
            List[np.ndarray]: 회전 정렬된 4x4 포즈 행렬 리스트
        """

        #  Fast3R 기준 카메라 위치 및 방향
        cam_position = np.array([-0.00141163, -0.01910395, -0.06794288], dtype=np.float32)
        cam_look_at  = np.array([-0.00352821, -0.01143425,  0.0154939],  dtype=np.float32)
        up_vector    = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # 기준 좌표계 계산: Z(forward), X(right), Y(up)
        forward = cam_look_at - cam_position
        forward /= np.linalg.norm(forward)

        right = np.cross(up_vector, forward)
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        # 회전 행렬: world → Fast3r 기준 좌표계
        R_align = np.stack([right, up, forward], axis=0)

        # 동차 변환 행렬 생성
        T_align = np.eye(4, dtype=np.float32)
        T_align[:3, :3] = R_align

        # 전체 포즈에 정렬 적용
        aligned_poses = []
        for pose in pose_list:
            aligned_pose = T_align @ pose
            aligned_poses.append(aligned_pose)

        return aligned_poses



    def is_front_view(self,
                      layout_mask: np.ndarray,
                      class_id: Optional[int] = None,
                      center_threshold: Optional[float] = None,
                      pixel_threshold: Optional[int] = None,
                      ) -> bool:
        """
        주어진 layout segmentation을 기반으로 이미지가 정면인지 판별

        Args:
            layout_mask (np.ndarray): (H, W) 형태의 세그멘테이션 마스크
            class_id (int): 정면 클래스 ID (기본: 1)
            center_threshold (float): 중심과의 거리 임계값 비율
            pixel_threshold (int): 유효 클래스로 인정할 최소 픽셀 수

        Returns:
            bool: 정면 뷰 여부 (True: 정면, False: 비정면)
        """

        if class_id is None:
            class_id = self.config.front_view_class_id
        if center_threshold is None:
            center_threshold = self.config.center_threshold
        if pixel_threshold is None:
            pixel_threshold = self.config.pixel_threshold

        # 모든 클래스의 픽셀 수 확인, pixel_threshold 미만인 클래스는 제외
        unique_classes = np.unique(layout_mask)
        valid_classes = []
        for cls in unique_classes:
            cls_mask = (layout_mask == cls).astype(np.uint8)
            cls_pixel_count = np.sum(cls_mask)
            if cls_pixel_count >= pixel_threshold:
                valid_classes.append(cls)

        # 유효 클래스(픽셀 수 ≥ pixel_threshold) 수가 5 미만이면 비정면
        if len(valid_classes) < 5:
            print('Layout 5장 이하')
            return False

        # 유효 클래스 수가 5이고 class_id=1이 유효 클래스에 포함된 경우, 중심 기반 정면 판단
        if class_id not in valid_classes:
            return False

        # 정면 클래스 컨투어 분석
        h, w = layout_mask.shape
        front_mask = (layout_mask == class_id).astype(np.uint8) * 255  # 컨투어 검출을 위해 255 스케일링
        contours, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # 가장 큰 컨투어 선택
        largest = max(contours, key=cv2.contourArea)
        x, y, w_rect, h_rect = cv2.boundingRect(largest)
        cx, cy = x + w_rect // 2, y + h_rect // 2
        dx, dy = abs(cx - w // 2), abs(cy - h // 2)
        return (dx < w * center_threshold) and (dy < h * center_threshold)

    @staticmethod
    def get_camera_position_and_direction(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        카메라 포즈로부터 위치(position)와 시점 방향(direction)을 추출

        Args:
            pose (np.ndarray): (4, 4) 형태의 카메라 extrinsic 행렬

        Returns:
            Tuple[np.ndarray, np.ndarray]: 카메라 위치, 정규화된 시점 벡터
        """

        position = pose[:3, 3]  # position: 행렬의 (0:3,3) 칼럼 / 반환 좌표계 기준으로
        direction = pose[:3, 2]  # irection: 행렬의 (0:3,2) 칼럼 (Z축) / 반환 좌표계 기준으로
        return position, direction / np.linalg.norm(direction)

    def compute_relative_angle(self,
                               pose1: np.ndarray,
                               pose2: np.ndarray,
                               ) -> float:
        """
        두 카메라 포즈 간 시점 벡터의 각도를 계산 (deg)

        Args:
            pose1 (np.ndarray): 첫 번째 카메라의 extrinsic 행렬
            pose2 (np.ndarray): 두 번째 카메라의 extrinsic 행렬

        Returns:
            float: 두 시점 간의 각도 (degree)
        """

        pos1, dir1 = self.get_camera_position_and_direction(pose1)
        pos2, dir2 = self.get_camera_position_and_direction(pose2)

        dir1_norm = dir1 / np.linalg.norm(dir1)
        dir2_norm = dir2 / np.linalg.norm(dir2)
        dot_product = np.dot(dir1_norm, dir2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def determine_relative_side(self,
                                pose1: np.ndarray,
                                pose2: np.ndarray,
                                ) -> str:
        """
        두 카메라의 상대적 위치를 기준으로 왼쪽 또는 오른쪽 방향 판별

        Args:
            pose1 (np.ndarray): 첫 번째 카메라 포즈
            pose2 (np.ndarray): 두 번째 카메라 포즈

        Returns:
            str: 'left' 또는 'right'
        """

        pos1, dir1 = self.get_camera_position_and_direction(pose1)
        pos2, dir2 = self.get_camera_position_and_direction(pose2)

        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)

        baseline = pos2 - pos1
        baseline_unit = baseline / np.linalg.norm(baseline)

        cross_vec = np.cross(dir1, dir2)
        direction_indicator = np.dot(cross_vec, baseline_unit)

        return 'left' if direction_indicator > 0 else 'right'

    @staticmethod
    def get_class_area(layout_seg: np.ndarray,
                       class_id: int,
                       ) -> int:
        """
        layout segmentation에서 주어진 클래스의 면적(픽셀 수)을 계산

        Args:
            layout_seg (np.ndarray): (H, W) 세그멘테이션 마스크
            class_id (int): 대상 클래스 ID

        Returns:
            int: 클래스의 픽셀 수
        """

        return np.sum(layout_seg == class_id)

    def decide_regeneration_from_angle_and_side(self,
                                                layout1: np.ndarray,
                                                layout2: np.ndarray,
                                                angle: float,
                                                side: str,
                                                z1: str,
                                                z2: str,
                                                ) -> Union[str, Tuple[str, str]]:
        """
        두 정면 이미지 간의 시점 각도 및 위치를 기반으로 재생성 이미지 방향을 결정(정밀 판별)

        Args:
            layout1 (np.ndarray): 첫 번째 정면 이미지의 layout segmentation
            layout2 (np.ndarray): 두 번째 정면 이미지의 layout segmentation
            angle (float): 두 이미지 간 시점 각도 (deg)
            side (str): 기준 이미지 기준 상대 방향 ('left' 또는 'right')
            z1 (str): 첫 번째 정면 이미지 이름
            z2 (str): 두 번째 정면 이미지 이름

        Returns:
            Union[str, Tuple[str, str]]: 'None' 또는 (선택 이미지명, 방향) 튜플

        """

        print(f"  → angle: {angle:.1f}°, side: {side}")

        # angle <=45 or >=135 → 거의 중첩이거나 정반대(180도)인 경우 (왼쪽 생성)
        if angle <= 45 or angle >= 135:
            print("   겹치거나 반대 시점이므로 → 왼쪽 생성")
            return (z1, 'right')

        print("   옆 시점이므로 → 사다리꼴(왼/오른쪽 벽) 넓이 비교 시작")

        # 그 외 옆 시점, 왼쪽/오른쪽 벽 넓이 비교 (더 정보가 많은 부분 생성)
        if side == 'right':
            area1 = self.get_class_area(layout1, class_id=2)  # z1 이미지에서 왼쪽 벽(class_id=2)
            area2 = self.get_class_area(layout2, class_id=3)  # z2 이미지에서 오른쪽 벽(class_id=3)
            print(f"    {z1}의 왼쪽 면적: {area1}, {z2}의 오른쪽 면적: {area2}")
            return (z1, 'left') if area1 > area2 else (z2, 'right')

        elif side == 'left':
            area1 = self.get_class_area(layout1, class_id=3)  # z1 이미지에서 오른쪽 벽
            area2 = self.get_class_area(layout2, class_id=2)  # z2 이미지에서 왼쪽 벽
            print(f"    {z1}의 오른쪽 면적: {area1}, {z2}의 왼쪽 면적: {area2}")
            return (z1, 'right') if area1 > area2 else (z2, 'left')

        print(f"   잘못된 side 값: {side}")
        return 'None'

    def process_images_with_pose(self,
                                 images: List[np.ndarray],
                                 poses: List[np.ndarray],
                                 ) -> List[Dict[str, Union[int, np.ndarray]]]:
        """
        전체 이미지에 대해 처리 수행:
        - 세그멘테이션 및 Theta 추론
        - 정면 이미지 판별
        - 정면 수에 따라 재생성 판단 수행

        Args:
            images (List[np.ndarray]): 입력 이미지 리스트 (3개)
            poses (List[np.ndarray]): 입력 포즈 리스트 (3개)

        Returns:
            List[Dict[str, Union[int, np.ndarray]]]: [{key: int, image: np.ndarray}, ...] 형식의 결과 리스트
        """

        front_views = []
        layout_map = {}  # {이미지명: layout_seg}
        theta_map = {}
        results = []

        for idx, img in enumerate(images):
            print(f"\n 처리 중: 이미지 {idx}")
            # 이미지 전처리
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.config.image_size)
            img_input = preprocess_input(img[tf.newaxis, ...])  # (1,400,400,3), float32

            # Layout Segmentation 예측 (stl 출력이 (1,400,400,1) 가정)
            layout_pred = self.model.predict(img_input)

            # layout_pred[0,:,:,0]는 실수형 예측 결과 → 반올림 후 uint8로 바꿔서 mask 생성
            layout_seg = np.rint(layout_pred[0, :, :, 0]).astype(np.uint8)
            layout_map[idx] = layout_seg

            # Theta 값 추출
            theta_values = self.theta_model.predict(img_input)[0]
            theta_map[idx] = theta_values

            # 정면 판별
            if self.is_front_view(layout_seg):
                front_views.append(idx)
                print(f"   > {idx}: 정면(True)")
            else:
                print(f"   > {idx}: 정면(False)")

        print(f"\n 최종 정면 이미지: {front_views}")

        # 정면 이미지 개수에 따른 재생성 판단
        # 1) 정면 이미지 3개 - 정보가 많아 재성성 불필요
        if len(front_views) >= 3:
            print("    정면 이미지가 3개 이상 → 재생성 불필요")
            return [{"key": 2, "image": None}]

        # 2) 정면 이미지 2개 - 정보가 부족한 부분 판단 후 재생성
        elif len(front_views) == 2:
            z1, z2 = front_views
            
            pose1 = poses[z1]
            pose2 = poses[z2]
            non_fronts = [i for i in range(3) if i not in front_views]  # 비정면 이미지에 대한 포즈

            # 정면 이미지 2개와 비정면 이미지 1개 간의 각도 비교
            related_angles = []
            for nf in non_fronts:
                nf_pose = poses[nf]
                angle1 = self.compute_relative_angle(pose1, nf_pose)
                angle2 = self.compute_relative_angle(pose2, nf_pose)
                related_angles.append((nf, angle1, angle2))

            for nf_name, angle1, angle2 in related_angles:
                print(f"    {nf_name}와의 각도 차이: {z1}: {angle1:.1f}°, {z2}: {angle2:.1f}°")

            # 두 정면 이미지 정보와 비정면 이미지의 정보가 겹치지 않는지 판별
            for nf_name, angle1, angle2 in related_angles:

                # 두 정면 이미지와 비정면 이미지 정보가 겹치지 않음
                if (angle1 <= 55 and angle2 <= 55) or (angle1 >= 90 and angle2 >= 90):
                    print("    두 정면 이미지 모두와 시점 차이가 크거나 매우 작음 → 정밀 판별 수행")
                    angle = self.compute_relative_angle(pose1, pose2)
                    side = self.determine_relative_side(pose1, pose2)
                    print(f"    두 정면 이미지 간 각도: {angle:.2f}°, 방향: {side}")
                    result = self.decide_regeneration_from_angle_and_side(
                        layout1=layout_map[z1],
                        layout2=layout_map[z2],
                        angle=angle,
                        side=side,
                        z1=str(z1),
                        z2=str(z2)
                    )
                    if result == 'None':
                        return [{"key": 2, "image": None}]
                    img_idx, direction = result
                    key = 0 if direction == 'left' else 1
                    return [{"key": key, "image": images[int(img_idx)]}]

                # 정면 이미지 중 하나의 정면 이미지만 정보가 겹침(겹치지 않는 정면 쪽 생성)
                elif angle1 <= 55 < angle2:
                    print(f"    이미지 {z2}와의 각도가 커서 상대적 방향으로 판단")
                    side = self.determine_relative_side(pose2, poses[nf_name])
                    key = 0 if side == 'left' else 1
                    return [{"key": key, "image": images[z2]}]
                elif angle2 <= 55 < angle1:
                    print(f"    이미지 {z1}와의 각도가 커서 상대적 방향으로 판단")
                    side = self.determine_relative_side(pose1, poses[nf_name])
                    key = 0 if side == 'left' else 1
                    return [{"key": key, "image": images[z1]}]

            print("    유효한 비교 대상 없음 → 기본 판별 수행")
            angle = self.compute_relative_angle(pose1, pose2)
            side = self.determine_relative_side(pose1, pose2)
            result = self.decide_regeneration_from_angle_and_side(
                layout1=layout_map[z1],
                layout2=layout_map[z2],
                angle=angle,
                side=side,
                z1=str(z1),
                z2=str(z2)
            )
            if result == 'None':
                return [{"key": 2, "image": None}]
            img_idx, direction = result
            key = 0 if direction == 'left' else 1
            return [{"key": key, "image": images[int(img_idx)]}]

        # 3) 정면 이미지 1개 - 부족한 정보 판단 후 재생성
        elif len(front_views) == 1:
            z_front = front_views[0]
            print(f"    정면 이미지가 1개: 이미지 {z_front}")

            non_fronts = [i for i in range(3) if i != z_front]
            if len(non_fronts) != 2:
                print("    정면이 1개지만 비교할 이미지가 부족 → 재생성 불가")
                return [{"key": 2, "image": None}]

            # 정면 이미지 1개와 비정면 이미지 2개의 위치 및 방향 비교
            pose_f = poses[z_front]
            pose_nf1 = poses[non_fronts[0]]
            pose_nf2 = poses[non_fronts[1]]

            _, z_dir1 = self.get_camera_position_and_direction(pose_nf1)
            _, z_dir2 = self.get_camera_position_and_direction(pose_nf2)

            z_dir1 = z_dir1 / np.linalg.norm(z_dir1)
            z_dir2 = z_dir2 / np.linalg.norm(z_dir2)

            dot_product = np.dot(z_dir1, z_dir2)
            print(f"    두 비정면 이미지 간 Z축 방향 유사도 (cosθ): {dot_product:.3f}")

            # 정면 이미지에 대해 비정면 이미지 2개가 같은 방향 - 반대 방향 재생성
            if dot_product > 0.8:
                pos_f, dir_f = self.get_camera_position_and_direction(pose_f)
                pos_nf1, _ = self.get_camera_position_and_direction(pose_nf1)
                side = self.determine_relative_side(pose_f, pose_nf1)
                opposite_side = 'right' if side == 'left' else 'left'
                print(f"   → 비정면 이미지 둘 다 {side} 방향에 있음 → 반대쪽 {opposite_side}로 생성")
                key = 0 if opposite_side == 'left' else 1
                return [{"key": key, "image": images[z_front]}]

            # 정면 이미지에 대해 비정면 이미지 2개가 다른 방향 - 정보 겹침으로 인한 재생성 불가
            print("   → 비정면 이미지 방향이 달라서 기준이 불분명 → 재생성 불가")
            return [{"key": 2, "image": None}]

        else:
            print("   → 정면 이미지 없음 → 재생성 불가")
            return [{"key": 2, "image": None}]

    def process(self, request_data, pose) -> List[Dict[str, Union[int, np.ndarray]]]:
        """
        모델을 사용하여 입력 데이터를 처리하고 결과를 반환

        Args:
            request_data: 이미지 데이터 (bytes 객체).
            pose (dict): 포즈 데이터 {'pose': [[...], [...], [...]]} 형식.

        Returns:
            List[Dict[str, Union[int, np.ndarray]]]: [{key: int, image: np.ndarray}, ...] 형식의 결과 리스트.
        """
        images, poses = self._load_data(request_data, pose)
        return self.process_images_with_pose(images, poses)

def dis_main(request_data, pose):
    """
    discriminator을 실행하고 결과를 직렬화

    Args:
        request_data: 입력 데이터 {image: [bytes, ...], pose: [list, ...]}.
        pose: 포즈 데이터 {'pose': [[...], [...], [...]]}.

    Returns:
        List[Dict[str, Union[int, np.ndarray]]]: [{key: int, image: np.ndarray | None}, ...] 형식의 결과 리스트.
    """
    start_time = time.time()
    
    processor = ShowRoomProcessor()
    result = processor.process(request_data, pose)
    
    elapsed = time.time() - start_time  # 경과 시간 계산
    print(f"\n discriminator 처리 시간: {elapsed:.2f}초")
    return result
