import os
import sys
sys.path.append('/content/drive/MyDrive/Final_Server/2d_server/ST-RoomNet')
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# TensorFlow Keras ConvNeXtTiny (TF 2.11 이상)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input

# spatial_transformer.py 에 정의된 ProjectiveTransformer 클래스
from spatial_transformer import ProjectiveTransformer


@dataclass
class ProcessorConfig:
    """프로세서 설정을 위한 데이터클래스"""
    
    # 입력 경로 설정
    val_path: str = '/content/drive/MyDrive/Final_Server/Input/Images'  # chrome extension에서 저장한 이미지
    pose_path: str = '/content/drive/MyDrive/Final_Server/Input/Poses/poses.txt'
    ref_img_path: str = '/content/drive/MyDrive/Final_Server/2d_server/ST-RoomNet/ref_img2.png'
    weight_path: str = '/content/drive/MyDrive/Final_Server/2d_server/ST-RoomNet/weights/Weight_ST_RroomNet_ConvNext.h5'
    
    # 출력 경로 설정
    save_path: str = '/content/drive/MyDrive/Final_Server/Input/Images'
    
    # 모델 설정
    image_size: Tuple[int, int] = (400, 400)
    input_channels: int = 3
    theta_dim: int = 8
    
    # 정면 판별 설정
    front_view_class_id: int = 1
    center_threshold: float = 0.25
    pixel_threshold: int = 5000
    
    # GPU 설정 (True: GPU 사용, False: CPU 사용)
    use_gpu: bool = False


class ShowRoomProcessor:
    """ShowRoom 레이아웃 처리를 위한 메인 클래스"""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        프로세서 초기화
        
        Args:
            config (Optional[ProcessorConfig]): 사용자 정의 설정. 없을 경우 기본 설정 사용
        """
        
        self.config = config if config is not None else ProcessorConfig()
        self.model = None
        self.theta_model = None
        self.img_names = []
        self.poses_map = {}
        
        self._setup_gpu()
        self._load_data()
        self._build_model()
    
    def _setup_gpu(self) -> None:
        """GPU 또는 CPU 사용 설정"""
        
        if not self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("CPU")
        else:
            print("GPU")
    
    def _load_data(self) -> None:
        """이미지 파일 및 포즈 데이터(.txt)를 로드하고 내부 구조로 매핑"""

        # 이미지 파일명 로드
        if not os.path.exists(self.config.val_path):
            raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다: {self.config.val_path}")

        img_filenames = sorted([f for f in os.listdir(self.config.val_path)
                                if f.endswith(('.jpg', '.png'))])
        self.img_names = [os.path.splitext(f)[0] for f in img_filenames]

        # 포즈 데이터 로드
        if not os.path.exists(self.config.pose_path):
            raise FileNotFoundError(f"포즈 파일을 찾을 수 없습니다: {self.config.pose_path}")

        with open(self.config.pose_path, "r") as f:
            pose_text = f.read()

        try:
            # 'array'와 'float32'만 정의해주는 안전한 eval 환경
            safe_env = {
                "array": np.array,
                "float32": np.float32,
            }
            pose_list = eval(pose_text, safe_env)
        except Exception as e:
            raise ValueError(f"포즈 파일 파싱 오류: {e}")

        if len(self.img_names) != len(pose_list):
            raise ValueError(f"이미지 수 ({len(self.img_names)})와 포즈 수 ({len(pose_list)})가 일치하지 않습니다")

        # 포즈 매핑
        self.poses_map = {name: pose for name, pose in zip(self.img_names, pose_list)}

        print(f" 데이터 로드 완료: 이미지 {len(self.img_names)}개, 포즈 {len(pose_list)}개")

    
    def _build_model(self) -> None:
        """ConvNeXt 기반 모델 구축 및 가중치(.h5) 로드 수행"""
        
        # 기준 이미지(ref_img2.png) 불러오기
        if not os.path.exists(self.config.ref_img_path):
            raise FileNotFoundError(f"{self.config.ref_img_path} 파일을 찾을 수 없습니다.")
        
        ref_img = tf.io.read_file(self.config.ref_img_path)
        ref_img = tf.io.decode_png(ref_img, channels=3)             # PNG를 RGB로 디코딩
        ref_img = tf.cast(ref_img, tf.float32) / 51.0               # 0~1 사이 정규화 (원래 코드 비율 유지)
        ref_img = tf.image.resize(ref_img, self.config.image_size)  # 크기 보정
        ref_img = ref_img[tf.newaxis, ...]                          # (1, 400, 400, 3) 배치 차원 추가
        
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
            bool: 정면(True) 여부
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

        # 유효 클래스(픽셀 수 ≥ pixel_threshold) 수가 5 미만이면 False
        if len(valid_classes) < 5:
            print('Layout 5장 이하')
            return False

        # 유효 클래스 수가 5이고 class_id=1이 유효 클래스에 포함된 경우, 중심 기반 정면 판단
        if class_id not in valid_classes:
            return False

        h, w = layout_mask.shape
        front_mask = (layout_mask == class_id).astype(np.uint8) * 255  # 컨투어 검출을 위해 255 스케일링
        contours, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # 가장 큰 컨투어 선택
        largest = max(contours, key=cv2.contourArea)
        x, y, w_rect, h_rect = cv2.boundingRect(largest)
        cx, cy = x + w_rect // 2, y + h_rect // 2
        dx, dy = abs(cx - w//2), abs(cy - h//2)
        return (dx < w * center_threshold) and (dy < h * center_threshold)

    @staticmethod
    def get_camera_position_and_direction(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        카메라 pose로부터 위치(position)와 시점 방향(direction)을 추출
        
        Args:
            pose (np.ndarray): (4, 4) 형태의 카메라 extrinsic 행렬
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 카메라 위치, 정규화된 시점 벡터
        """
        
        position = pose[:3, 3]  # position: 행렬의 (0:3,3) 칼럼 / 반환 좌표계 기준으로
        direction = pose[:3, 2] # irection: 행렬의 (0:3,2) 칼럼 (Z축) / 반환 좌표계 기준으로
        return position, direction / np.linalg.norm(direction)

    def compute_relative_angle(self,
                               pose1: np.ndarray,
                               pose2: np.ndarray,
                               ) -> float:
        """
        두 카메라 pose 간의 시점 벡터 간 각도를 계산 (deg)

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
        두 정면 이미지 간의 시점 각도 및 위치를 기반으로 재생성 이미지 방향을 결정

        Args:
            layout1 (np.ndarray): 첫 번째 이미지의 layout segmentation
            layout2 (np.ndarray): 두 번째 이미지의 layout segmentation
            angle (float): 두 이미지 간 시점 각도 (deg)
            side (str): 기준 이미지 기준 상대 방향 ('left' 또는 'right')
            z1 (str): 첫 번째 이미지 이름
            z2 (str): 두 번째 이미지 이름

        Returns:
            Union[str, Tuple[str, str]]: 'both' 또는 (선택 이미지명, 방향) 튜플

        """
        
        print(f"  → angle: {angle:.1f}°, side: {side}")

        # 1) 거의 중첩이거나 정반대(180도)인 경우
        # angle <=45 or >=135 → 두 이미지가 거의 동일 정면이거나 정반대 뷰 → 양쪽 모두 재생성
        if angle <= 45 or angle >= 135:
            print("   겹치거나 반대 시점이므로 → 양쪽 모두 재생성")
            return 'both'

        print("   옆 시점이므로 → 사다리꼴(왼/오른쪽 벽) 넓이 비교 시작")
        
        # 2) 그 외(가로 회전 시점)
        # 왼쪽/오른쪽 벽 넓이 비교해 어느 쪽을 쓸지 결정
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
        return None

    def process_images_with_pose(self) -> Union[str, Tuple[str, str], None]:
        """
        전체 이미지에 대해 처리 수행:
        - 세그멘테이션 및 Theta 추론
        - 정면 이미지 판별
        - 정면 수에 따라 재생성 판단 수행

        Returns:
            Union[str, Tuple[str, str], None]: 재생성 판단 결과
        """
        
        os.makedirs(self.config.save_path, exist_ok=True)
        front_views = []
        layout_map = {}    # {이미지명: layout_seg}
        theta_map = {}

        for img_name in self.img_names:
            print(f"\n 처리 중: {img_name}.jpg")
            img_path = os.path.join(self.config.val_path, img_name + '.jpg')
            if not os.path.exists(img_path):
                print(f"   {img_path}을(를) 찾을 수 없음 → 건너뜀")
                continue

            # 이미지 읽고 RGB 전처리
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.config.image_size)
            img_input = preprocess_input(img[tf.newaxis, ...])  # (1,400,400,3), float32

            # Layout Segmentation 예측 (stl 출력이 (1,400,400,1) 가정)
            layout_pred = self.model.predict(img_input)
            
            # layout_pred[0,:,:,0]는 실수형 예측 결과 → 반올림 후 uint8로 바꿔서 mask 생성
            layout_seg = np.rint(layout_pred[0, :, :, 0]).astype(np.uint8)
            layout_map[img_name] = layout_seg

            # Theta 값 추출
            theta_values = self.theta_model.predict(img_input)[0]
            theta_map[img_name] = theta_values
            
            # θ를 파일로 저장
            np.savez(os.path.join(self.config.save_path, f'{img_name}_theta.npz'), theta=theta_values)
            
            # Segmentation mask를 시각화(51*레이블)하여 PNG로 저장
            cv2.imwrite(os.path.join(self.config.save_path, f'{img_name}_pred.png'), layout_seg * 51)

            # 정면 판별
            if self.is_front_view(layout_seg):
                front_views.append(img_name)
                print(f"   > {img_name}: 정면(True)")
            else:
                print(f"   > {img_name}: 정면(False)")

        print(f"\n 최종 정면 이미지: {front_views}")

        # 정면 이미지 개수에 따른 재생성 판단
        if len(front_views) >= 3:
            print("   → 정면 이미지가 3개 이상 → 재생성 불필요")
            return None

        elif len(front_views) == 2:
            z1, z2 = front_views
            pose1 = self.poses_map[z1]
            pose2 = self.poses_map[z2]
            angle = self.compute_relative_angle(pose1, pose2)
            side = self.determine_relative_side(pose1, pose2)
            print(f"   → 두 정면 이미지 간 각도: {angle:.2f}°, 방향: {side}")
            return self.decide_regeneration_from_angle_and_side(
                layout1=layout_map[z1],
                layout2=layout_map[z2],
                angle=angle,
                side=side,
                z1=z1,
                z2=z2
            )

        elif len(front_views) == 1:
            print("   → 정면 이미지가 1개 → 양쪽 모두 생성 고려")
            return 'both'

        else:
            print("   → 정면 이미지 없음 → 전체 재생성 필요")
            return 'both'

    def save_result(self, decision: Union[str, Tuple[str, str], None]) -> str:
        """
        처리 결과를 텍스트 파일(.txt)로 저장

        Args:
            decision (Union[str, Tuple[str, str], None]): 재생성 판단 결과

        Returns:
            str: 저장된 파일 경로
        """
        
        output_txt_path = os.path.join(self.config.save_path, "ST_result.txt")

        with open(output_txt_path, "w") as f:
            
            # 정면 0개 또는 1개인 경우, 모든 이미지에 대해 '2' 처리
            if decision == 'both':           
                for name in self.img_names:
                    f.write(f"{name} 2\n")
            elif isinstance(decision, tuple):
                selected_img, side = decision
                side_code = {'left': 0, 'right': 1}.get(side, 2)
                f.write(f"{selected_img} {side_code}\n")
                
            # 예외 처리 (None 등)
            else:                           
                f.write("none 2\n")

        print(f"\n 결과 텍스트 저장 완료: {output_txt_path}")
        return output_txt_path

    def process(self) -> Union[str, Tuple[str, str], None]:
        """
        전체 프로세스를 실행하는 메인 메서드
        
        - 이미지 전처리 및 분석 수행
        - 결과 판단 및 저장
        
        Returns:
            Union[str, Tuple[str, str], None]: 최종 재생성 판단 결과 / ('both', (이미지명, 방향), None)
        """
        
        print("ShowRoom 레이아웃 처리를 시작합니다...")
        
        # 이미지 처리 및 분석
        decision = self.process_images_with_pose()
        
        # 결과 저장
        self.save_result(decision)
        
        print(f"\n 최종 재생성 판단 결과: {decision}")
        return decision


def main():
    """
    메인 실행 함수

    스크립트 실행 시 ShowRoomProcessor를 실행하고 결과를 출력함
    """
    
    # 기본 설정으로 프로세서 생성 및 실행
    processor = ShowRoomProcessor()
    result = processor.process()
    return result


if __name__ == "__main__":
    # 스크립트로 직접 실행할 때만 동작
    main()
