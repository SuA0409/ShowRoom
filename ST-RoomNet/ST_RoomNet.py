# test.py 코드 복붙
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval

# TensorFlow Keras ConvNeXtTiny (TF 2.11 이상)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input

# spatial_transformer.py 에 정의된 ProjectiveTransformer 클래스
from spatial_transformer import ProjectiveTransformer

# ─────────────────────────────────────────────────────────────────────────────
# 2) Input
# ─────────────────────────────────────────────────────────────────────────────

# image 불러오기
val_path = '/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Images'
img_filenames = sorted([f for f in os.listdir(val_path) if f.endswith(('.jpg', '.png'))])
img_names = [os.path.splitext(f)[0] for f in img_filenames]  # ['0', '1', '2', ...]


# pose 불러오기
pose_path = '/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Pose/pose.txt'
with open(pose_path, 'r') as f:
    pose_text = f.read()

# 문자열을 실제 리스트로 변환
pose_list = eval(pose_text, {"array": np.array, "float32": np.float32})

# numpy array로 변환
pose_list = [np.array(p, dtype=np.float32) for p in pose_list]

# 이미지 이름과 포즈 수가 같아야 함
print (len(img_names),len(pose_list))
assert len(img_names) == len(pose_list), "이미지 수와 포즈 수가 일치하지 않습니다"



# 포즈 매핑
poses_map = {name: pose for name, pose in zip(img_names, pose_list)}
# ─────────────────────────────────────────────────────────────────────────────
# 3) GPU 설정 (필요시 수정)
# ─────────────────────────────────────────────────────────────────────────────

# Colab에서 GPU를 사용하려면 이 줄을 주석 처리 또는 삭제하세요.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ─────────────────────────────────────────────────────────────────────────────
# 4) 모델 로드 및 가중치 설정
# ─────────────────────────────────────────────────────────────────────────────

# 4-1) 기준 이미지(ref_img2.png) 불러오기
#      Colab 환경에 'ref_img2.png' 를 업로드해두거나, Drive와 연동해 경로 지정해 주세요.
ref_img_path = 'ref_img2.png'
if not os.path.exists(ref_img_path):
    raise FileNotFoundError(f"{ref_img_path} 파일을 Colab 환경에 업로드하셨는지 확인하세요.")

ref_img = tf.io.read_file(ref_img_path)
ref_img = tf.io.decode_png(ref_img, channels=3)           # PNG를 RGB로 디코딩
ref_img = tf.cast(ref_img, tf.float32) / 51.0             # 0~1 사이 정규화 (원래 코드 비율 유지)
ref_img = tf.image.resize(ref_img, (400, 400))            # 크기 보정
ref_img = ref_img[tf.newaxis, ...]                        # (1, 400, 400, 3) 배치 차원 추가

# 4-2) ConvNeXtTiny Base 모델 (include_top=False, pooling='avg')
#      input_shape=(400,400,3)으로 지정
base_model = ConvNeXtTiny(
    include_top=False,
    weights="imagenet",
    input_shape=(400, 400, 3),
    pooling='avg'
)

# 4-3) Theta 값을 예측할 Dense 레이어 추가
theta_layer = Dense(8, name='theta_layer')(base_model.output)

# 4-4) ProjectiveTransformer로 Warping 수행
#      - (400,400) 출력 크기를 ProjectiveTransformer 생성자에 전달
transformer = ProjectiveTransformer((400, 400))

#    stl: Spatial Transformer Layer의 출력 (정규화되지 않은 형태)
#       입력 이미지는 ref_img(고정) → theta 값은 trainable
stl = transformer.transform(ref_img, theta_layer)

# 4-5) 최종 모델: 입력 → Theta → Spatial Transformer 변환 출력
model = Model(inputs=base_model.input, outputs=stl)

# 4-6) 사전에 학습된 가중치(.h5) 불러오기
#      Colab 환경에 Weight_ST_RroomNet_ConvNext.h5 를 업로드하거나 Drive 연동 후 경로 수정하세요.
weight_path = '/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/ST-RoomNet/weights/Weight_ST_RroomNet_ConvNext.h5'
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"{weight_path} 파일을 Colab 환경에 업로드하셨는지 확인하세요.")

model.load_weights(weight_path)
print(" 메인 모델 가중치 로드 완료")

# 4-7) Theta만 별도로 뽑아내기 위한 서브 모델
theta_model = Model(inputs=base_model.input, outputs=theta_layer)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Helper 함수 정의
# ─────────────────────────────────────────────────────────────────────────────

def is_front_view(layout_mask, class_id=1, center_threshold=0.25):
    """
    layout_mask: (H,W) uint8 형태의 세그멘테이션 레이블
    class_id   : 정면(전벽)에 해당하는 클래스 ID
    center_threshold: 중심 기준 판별 비율(이미지 폭/높이 대비)

    정면(view)인지 판단하려면,
    mask에서 class_id 위치를 찾아 BoundingBox의 중심이 이미지 중심에 가까운지 확인.
    """
    h, w = layout_mask.shape
    front_mask = (layout_mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # 가장 큰 컨투어 선택
    largest = max(contours, key=cv2.contourArea)
    x, y, w_rect, h_rect = cv2.boundingRect(largest)
    cx, cy = x + w_rect // 2, y + h_rect // 2
    dx, dy = abs(cx - w//2), abs(cy - h//2)
    return (dx < w * center_threshold) and (dy < h * center_threshold)

def get_camera_position_and_direction(pose):
    """
    pose: (4,4) 카메라 extrinsic 행렬
    반호계 좌표계 기준으로:
    - position: 행렬의 (0:3,3) 칼럼
    - direction: 행렬의 (0:3,2) 칼럼 (Z축)
    """
    position = pose[:3, 3]
    direction = pose[:3, 2]
    return position, direction / np.linalg.norm(direction)

def compute_relative_angle(pose1, pose2):
    """
    두 카메라 pose 간의 시점 벡터 간 각도를 계산 (deg)
    """
    pos1, dir1 = get_camera_position_and_direction(pose1)
    pos2, dir2 = get_camera_position_and_direction(pose2)

    dir1_norm = dir1 / np.linalg.norm(dir1)
    dir2_norm = dir2 / np.linalg.norm(dir2)
    dot_product = np.dot(dir1_norm, dir2_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def determine_relative_side(pose1, pose2):
    """
    두 카메라 pose 간의 시점 방향이 왼쪽/오른쪽 어느 쪽인지 판별
    """
    pos1, dir1 = get_camera_position_and_direction(pose1)
    pos2, dir2 = get_camera_position_and_direction(pose2)

    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)

    baseline = pos2 - pos1
    baseline_unit = baseline / np.linalg.norm(baseline)

    cross_vec = np.cross(dir1, dir2)
    direction_indicator = np.dot(cross_vec, baseline_unit)

    return 'left' if direction_indicator > 0 else 'right'

def get_class_area(layout_seg, class_id):
    """
    layout_seg: (H,W) ndarray, class_id별 픽셀 개수(면적) 계산
    """
    return np.sum(layout_seg == class_id)

def decide_regeneration_from_angle_and_side(layout1, layout2, angle, side, z1, z2):
    """
    angle: 두 시점 간의 각도 (deg)
    side : 'left' or 'right'

    - angle <=45 or >=135 → 두 이미지가 거의 동일 정면이거나 정반대 뷰 → 양쪽 모두 재생성
    - 그 외(가로 회전 시점) → 왼쪽/오른쪽 벽 넓이 비교해 어느 쪽을 쓸지 결정
    """
    print(f"  → angle: {angle:.1f}°, side: {side}")

    # 1) 거의 중첩이거나 정반대(180도)인 경우
    if angle <= 45 or angle >= 135:
        print("   겹치거나 반대 시점이므로 → 양쪽 모두 재생성")
        return 'both'

    print("   옆 시점이므로 → 사다리꼴(왼/오른쪽 벽) 넓이 비교 시작")
    if side == 'right':
        area1 = get_class_area(layout1, class_id=2)  # z1 이미지에서 왼쪽 벽(class_id=2)
        area2 = get_class_area(layout2, class_id=3)  # z2 이미지에서 오른쪽 벽(class_id=3)
        print(f"    {z1}의 왼쪽 면적: {area1}, {z2}의 오른쪽 면적: {area2}")
        return (z1, 'left') if area1 > area2 else (z2, 'right')

    elif side == 'left':
        area1 = get_class_area(layout1, class_id=3)  # z1 이미지에서 오른쪽 벽
        area2 = get_class_area(layout2, class_id=2)  # z2 이미지에서 왼쪽 벽
        print(f"    {z1}의 오른쪽 면적: {area1}, {z2}의 왼쪽 면적: {area2}")
        return (z1, 'right') if area1 > area2 else (z2, 'left')

    print(f"   잘못된 side 값: {side}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 6) 메인 처리 함수 정의
# ─────────────────────────────────────────────────────────────────────────────

def process_images_with_pose(
    img_names,         # 예: ['000000','000001','000002']
    poses_map,         # {'000000':pose1, '000001':pose2, '000002':pose3}
    model,             # Spatial Transformer Model
    theta_model,       # Theta만 추출하는 모델
    val_path,     # 원본 이미지(.jpg) 폴더 경로
    save_path  # 출력(예측 결과) 저장 폴더
):
    os.makedirs(save_path, exist_ok=True)
    front_views = []
    layout_map = {}    # {이미지명: layout_seg}
    theta_map = {}

    for img_name in img_names:
        print(f"\n▶ 처리 중: {img_name}.jpg")
        img_path = os.path.join(val_path, img_name + '.jpg')
        if not os.path.exists(img_path):
            print(f"   ⚠ {img_path}을(를) 찾을 수 없음 → 건너뜀")
            continue

        # 6-1) 이미지 읽고 RGB 전처리
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))
        img_input = preprocess_input(img[tf.newaxis, ...])  # (1,400,400,3), float32

        # 6-2) Layout Segmentation 예측 (stl 출력이 (1,400,400,1) 가정)
        layout_pred = model.predict(img_input)
        # layout_pred[0,:,:,0]는 실수형 예측 결과 → 반올림 후 uint8로 바꿔서 mask 생성
        layout_seg = np.rint(layout_pred[0, :, :, 0]).astype(np.uint8)
        layout_map[img_name] = layout_seg

        # 6-3) Theta 값 추출
        theta_values = theta_model.predict(img_input)[0]
        theta_map[img_name] = theta_values
        # θ를 파일로 저장
        np.savez(os.path.join(save_path, f'{img_name}_theta.npz'), theta=theta_values)
        # Segmentation mask를 시각화(51*레이블)하여 PNG로 저장
        cv2.imwrite(os.path.join(save_path, f'{img_name}_pred.png'), layout_seg * 51)

        # 6-4) 정면 판별
        if is_front_view(layout_seg, class_id=1, center_threshold=0.25):
            front_views.append(img_name)
            print(f"   > {img_name}: 정면(True)")
        else:
            print(f"   > {img_name}: 정면(False)")

    print(f"\n▶ 최종 정면 이미지: {front_views}")

    # 6-5) 정면 이미지 개수에 따른 재생성 판단
    if len(front_views) >= 3:
        print("   → 정면 이미지가 3개 이상 → 재생성 불필요")
        return None

    elif len(front_views) == 2:
        z1, z2 = front_views
        pose1 = poses_map[z1]
        pose2 = poses_map[z2]
        angle = compute_relative_angle(pose1, pose2)
        side = determine_relative_side(pose1, pose2)
        print(f"   → 두 정면 이미지 간 각도: {angle:.2f}°, 방향: {side}")
        return decide_regeneration_from_angle_and_side(
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

# ─────────────────────────────────────────────────────────────────────────────
# 7) 메인 함수 실행
# ─────────────────────────────────────────────────────────────────────────────

# ※ val_path에 실제로 .jpg 파일들이 존재해야 합니다.
#    예: '/content/drive/MyDrive/images/' 등으로 수정 가능
save_path = '/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/ST'

decision = process_images_with_pose(
    img_names=img_names,
    poses_map=poses_map,
    model=model,
    theta_model=theta_model,
    val_path=val_path,
    save_path=save_path
)
print(f"\n▶ 최종 재생성 판단 결과: {decision}")
# ─────────────────────────────────────────────────────────────────────────────
# 8) 결과 저장
# ─────────────────────────────────────────────────────────────────────────────

output_txt_path = os.path.join(save_path, "ST_result.txt")

with open(output_txt_path, "w") as f:
    if decision == 'both':
        # 정면 0개 또는 1개인 경우, 모든 이미지에 대해 '2' 처리
        for name in img_names:
            f.write(f"{name} 2\n")
    elif isinstance(decision, tuple):
        selected_img, side = decision
        side_code = {'left': 0, 'right': 1}.get(side, 2)
        f.write(f"{selected_img} {side_code}\n")
    else:
        # 예외 처리 (None 등)
        f.write("none 2\n")

print(f"\n▶ 결과 텍스트 저장 완료: {output_txt_path}")