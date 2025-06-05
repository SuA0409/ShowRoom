import os
import shutil
import pandas as pd
import random
import math

## 설정
SRC_ROOT = r'E:\scannet\data_download\data'  # 원본 ScanNet 데이터 경로
DST_ROOT = r'C:\Users\최낙민.DESKTOP-RHND68Q\Desktop\scannet_test'  # 복사 및 샘플링 결과 저장 경로
IMAGE_SUBDIR = ['dslr', 'resized_undistorted_images']  # 이미지가 저장된 하위 폴더 구조
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')  # 대상 이미지 확장자
SAMPLE_IMAGE_COUNT = 5  # 샘플당 이미지 수
NUM_SETS_PER_STRATEGY = 10  # 전략별 세트 수
STRATEGIES = ['sequential', 'gap5', 'gap10', 'gap20', 'gap30', 'gap40']  # 샘플링 간격
FOLDER_NAMES = ['image_min', 'image_avg', 'image_max']  # 결과 저장용 폴더명
PIXEL_SORT_KEY = lambda x: int(''.join(filter(str.isdigit, x)))

## 함수
# 방별 이미지 수 계산
def get_room_image_counts(src_root):
    room_dirs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    room_image_counts = []
    for room in room_dirs:
        image_dir = os.path.join(src_root, room, *IMAGE_SUBDIR)
        if os.path.exists(image_dir):
            image_count = len([
                f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
            ])
        else:
            image_count = 0
        room_image_counts.append({'room': room, 'image_count': image_count})
    return pd.DataFrame(room_image_counts)

# 최소, 평균, 최대 방 이름 반환
def get_target_rooms(df):
    min_room = df.loc[df['image_count'].idxmin()]['room']
    max_room = df.loc[df['image_count'].idxmax()]['room']
    mean_val = df['image_count'].mean()
    mean_room = df.iloc[(df['image_count'] - mean_val).abs().argsort()[:1]]['room'].values[0]
    return {'image_min': min_room, 'image_avg': mean_room, 'image_max': max_room}

# 지정된 방 이미지 복사
def copy_room_images(src_root, dst_root, folder_room_map):
    for folder, room in folder_room_map.items():
        src = os.path.join(src_root, room, *IMAGE_SUBDIR)
        dst = os.path.join(dst_root, folder)
        os.makedirs(dst, exist_ok=True)
        if os.path.exists(src):
            for fname in sorted(os.listdir(src), key=PIXEL_SORT_KEY):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))

# 샘플 인덱스 생성
def get_sampling_indices(start, gap, total, num_samples=SAMPLE_IMAGE_COUNT):
    return [min(start + i * gap, total - 1) for i in range(num_samples)]

# 샘플링
def sample_images(base_path, folders, strategies, num_sets):
    sampling_path = os.path.join(base_path, 'sampling')
    os.makedirs(sampling_path, exist_ok=True)

    for folder in folders:
        image_folder = os.path.join(base_path, folder)
        image_files = sorted([
            f for f in os.listdir(image_folder) if f.lower().endswith(IMAGE_EXTENSIONS)
        ], key=PIXEL_SORT_KEY)
        total_images = len(image_files)

        for strategy in strategies:
            gap = 1 if strategy == 'sequential' else max(1, math.floor(total_images / int(strategy.replace('gap', ''))))

            for set_idx in range(num_sets):
                max_start = total_images - gap * (SAMPLE_IMAGE_COUNT - 1)
                if max_start <= 0:
                    continue
                start_idx = random.randint(0, max_start)
                indices = get_sampling_indices(start_idx, gap, total_images)
                selected_files = [image_files[i] for i in indices]

                save_dir = os.path.join(sampling_path, folder, strategy, f'set_{set_idx + 1}')
                os.makedirs(save_dir, exist_ok=True)

                for fname in selected_files:
                    src_path = os.path.join(image_folder, fname)
                    dst_path = os.path.join(save_dir, fname)
                    shutil.copy2(src_path, dst_path)

## 실행
def run_scannet_sampling():
    # 1. 이미지 수 기준 대상 방 선정
    df = get_room_image_counts(SRC_ROOT)
    folder_room_map = get_target_rooms(df)

    # 2. 대상 방 이미지 복사
    copy_room_images(SRC_ROOT, DST_ROOT, folder_room_map)

    # 3. 샘플링 수행
    sample_images(DST_ROOT, FOLDER_NAMES, STRATEGIES, NUM_SETS_PER_STRATEGY)

    print("전체 작업 완료: 방 복사 및 샘플링")

# 실행
if __name__ == '__main__':
    run_scannet_sampling()
