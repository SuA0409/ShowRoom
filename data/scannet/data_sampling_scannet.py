import os
import shutil
import re

## 경로 및 변수 설정
SRC_ROOT = r'E:\scannet\data_download\data'
BASE_DST_ROOT = r'G:\내 드라이브\Scannet++\data_scannet_fin'
DSX_ROOT = r'G:\내 드라이브\Scannet++'
TXT_PATH = os.path.join(DSX_ROOT, 'processed_rooms_4.txt')
MIN_IMAGE_COUNT = 20        # 최소 이미지 개수 조건
SET_IMAGE_COUNT = 5         # 한 세트에 포함될 이미지 개수
FOLDER_SPLIT_SIZE = 4000    # 폴더를 나눌 기준 개수 (4000개 단위로 새 폴더 생성)

## 유틸 함수
# 파일명에서 숫자 추출
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# 처리된 방 목록 불러오기
def load_processed_rooms(txt_path):
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

# 처리 완료된 방 텍스트 파일에 저장(중복 방지)
def save_processed_room(txt_path, room):
    with open(txt_path, 'a') as f:
        f.write(f"{room}\n")

# 새 결과 저장 폴더
def ensure_dst_root(base_dst_root, current_upper_idx):
    dst_root = os.path.join(base_dst_root, f'data_scannet_r_{current_upper_idx}')
    os.makedirs(dst_root, exist_ok=True)
    return dst_root

# 이미지 파일 복사
def copy_images(image_dir, image_files, indices, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for idx in indices:
        file = image_files[idx]
        shutil.copy(os.path.join(image_dir, file), os.path.join(dst_folder, file))

## 메인 함수
# 전체 방 폴더 순회, 이미지 샘플링 및 복사
def process_rooms(src_root, base_dst_root, txt_path):
    processed_rooms = load_processed_rooms(txt_path)
    room_folders = sorted(os.listdir(src_root))

    current_upper_idx = 10
    dst_root = ensure_dst_root(base_dst_root, current_upper_idx)

    # 현재 폴더 내 가장 높은 번호를 인덱스로 설정
    existing_folders = [name for name in os.listdir(dst_root) if name.isdigit()]
    folder_idx = max([int(name) for name in existing_folders], default=0) + 1

    for room in room_folders:
        if room in processed_rooms:
            continue

        image_dir = os.path.join(src_root, room, 'dslr', 'resized_undistorted_images')
        if not os.path.isdir(image_dir):
            continue

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(image_files)

        if total < MIN_IMAGE_COUNT:
            print(f" {room} 폴더는 이미지가 부족하여 건너뜀 ({total}장)")
            continue

        # 샘플링을 위한 간격    
        k = total // SET_IMAGE_COUNT
        num_sets = total // (SET_IMAGE_COUNT * 4)
        if num_sets == 0:
            continue
        p = k // num_sets

        # 이미지 샘플링
        for j in range(p):
            base_idx = j * (k + num_sets)
            for i in range(num_sets):
                base_idx += 1
                indices = [base_idx + t * num_sets for t in range(SET_IMAGE_COUNT)]

                if indices[-1] >= total:
                    continue

                dst_folder = os.path.join(dst_root, str(folder_idx))
                copy_images(image_dir, image_files, indices, dst_folder)

                # 지정된 개수마다 상위 폴더 인덱스 증가
                if folder_idx % FOLDER_SPLIT_SIZE == 0:
                    current_upper_idx += 1
                    dst_root = ensure_dst_root(base_dst_root, current_upper_idx)

                folder_idx += 1

        save_processed_room(txt_path, room)
        print(f" {room} 완료: {folder_idx}번까지 생성됨")

    print("모든 작업 완료!")

## 실행
if __name__ == '__main__':
    process_rooms(SRC_ROOT, BASE_DST_ROOT, TXT_PATH)
