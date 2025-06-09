import os
import shutil
import re


class ScannetRoomProcessor:
    def __init__(self, src_root, dst_root_base, txt_path,
                 min_image_count=20, set_image_count=5, folder_split_size=4000):
        self.src_root = src_root
        self.dst_root_base = dst_root_base
        self.txt_path = txt_path
        self.min_image_count = min_image_count        # 최소 이미지 개수 조건
        self.set_image_count = set_image_count        # 한 세트에 포함될 이미지 개수
        self.folder_split_size = folder_split_size    # 폴더를 나눌 기준 개수 (4000개 단위로 새 폴더 생성)
        self.processed_rooms = self._load_processed_rooms()
        self.current_upper_idx = 10                   # 저장 폴더 인덱스 시작값
        self.dst_root = self._ensure_dst_root()       # 현재 상위 저장 폴더 경로
        self.folder_idx = self._get_initial_folder_idx()  # 하위 폴더 시작 인덱스

    # 처리된 방 목록 불러오기
    def _load_processed_rooms(self):
        if os.path.exists(self.txt_path):
            with open(self.txt_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    # 처리 완료된 방 텍스트 파일에 저장(중복 방지)
    def _save_processed_room(self, room):
        with open(self.txt_path, 'a') as f:
            f.write(f"{room}\n")

    # 새 결과 저장 폴더
    def _ensure_dst_root(self):
        dst_root = os.path.join(self.dst_root_base, f'data_scannet_r_{self.current_upper_idx}')
        os.makedirs(dst_root, exist_ok=True)
        return dst_root

    # 현재 폴더 내 가장 높은 번호를 인덱스로 설정
    def _get_initial_folder_idx(self):
        existing = [name for name in os.listdir(self.dst_root) if name.isdigit()]
        return max([int(name) for name in existing], default=0) + 1

    # 이미지 파일 복사
    def _copy_images(self, image_dir, image_files, indices, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for idx in indices:
            src_file = os.path.join(image_dir, image_files[idx])
            dst_file = os.path.join(dst_folder, image_files[idx])
            shutil.copy(src_file, dst_file)

    # 파일명에서 숫자 추출
    def _extract_number(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1

    # 전체 방 폴더 순회, 이미지 샘플링 및 복사
    def process_all(self):
        for room in sorted(os.listdir(self.src_root)):
            if room in self.processed_rooms:
                continue
            self._process_room(room)
        print("모든 작업 완료!")

    # 개별 방 처리
    def _process_room(self, room):
    image_dir = os.path.join(self.src_root, room, 'dslr', 'resized_undistorted_images')
    if not os.path.isdir(image_dir):
        return

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=self._extract_number
    )
    total_images = len(image_files)

    if total_images < self.min_image_count:
        print(f" {room} 폴더는 이미지가 부족하여 건너뜀 ({total_images}장)")
        return

    # 샘플링 간격: 한 세트를 구성할 수 있는 최소 간격 계산
    sampling_interval = total_images // (self.set_image_count * 4)
    if sampling_interval == 0:
        return

    # 한 간격당 이미지 5장씩 건너뛰며 뽑을 수 있는 반복 횟수
    total_span = total_images // self.set_image_count
    num_iterations = total_span // sampling_interval

    for iter_idx in range(num_iterations):
        start_base = iter_idx * (sampling_interval + total_span)
        for offset in range(sampling_interval):
            base_idx = start_base + offset + 1  # 시작점 보정

            indices = [
                base_idx + t * sampling_interval for t in range(self.set_image_count)
            ]
            if indices[-1] >= total_images:
                continue

            dst_folder = os.path.join(self.dst_root, str(self.folder_idx))
            self._copy_images(image_dir, image_files, indices, dst_folder)

            if self.folder_idx % self.folder_split_size == 0:
                self.current_upper_idx += 1
                self.dst_root = self._ensure_dst_root()

            self.folder_idx += 1

    self._save_processed_room(room)
    print(f" {room} 완료: {self.folder_idx}번까지 생성됨")


if __name__ == '__main__':
    processor = ScannetRoomProcessor(
        src_root=r'E:\scannet\data_download\data',
        dst_root_base=r'G:\내 드라이브\Scannet++\data_scannet_fin',
        txt_path=r'G:\내 드라이브\Scannet++\processed_rooms_4.txt'
    )
    processor.process_all()
