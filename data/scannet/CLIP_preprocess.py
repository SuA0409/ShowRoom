import os
import shutil
import pandas as pd
import random
import math

class ScannetSampler:
    ## 설정
    def __init__(
        self,
        src_root,
        dst_root,
        image_subdir=None,
        image_extensions=None,
        sample_image_count=5,
        num_sets_per_strategy=10,
        strategies=None,
        folder_names=None
    ):
        self.src_root = src_root  # 원본 ScanNet 데이터 경로
        self.dst_root = dst_root  # 복사 및 샘플링 결과 저장 경로
        self.image_subdir = image_subdir or ['dslr', 'resized_undistorted_images']  # 이미지가 저장된 하위 폴더 구조
        self.image_extensions = image_extensions or ('.jpg', '.jpeg', '.png')  # 대상 이미지 확장자
        self.sample_image_count = sample_image_count  # 샘플당 이미지 수
        self.num_sets_per_strategy = num_sets_per_strategy  # 전략별 세트 수
        self.strategies = strategies or ['sequential', 'gap5', 'gap10', 'gap20', 'gap30', 'gap40']  # 샘플링 간격
        self.folder_names = folder_names or ['image_min', 'image_avg', 'image_max']  # 결과 저장용 폴더명
        self.pixel_sort_key = lambda x: int(''.join(filter(str.isdigit, x)))

    # 방별 이미지 수 계산
    def get_room_image_counts(self):
        room_dirs = [d for d in os.listdir(self.src_root) if os.path.isdir(os.path.join(self.src_root, d))]
        room_image_counts = []
        for room in room_dirs:
            image_dir = os.path.join(self.src_root, room, *self.image_subdir)
            if os.path.exists(image_dir):
                image_count = len([
                    f for f in os.listdir(image_dir) if f.lower().endswith(self.image_extensions)
                ])
            else:
                image_count = 0
            room_image_counts.append({'room': room, 'image_count': image_count})
        return pd.DataFrame(room_image_counts)

    # 최소, 평균, 최대 방 이름 반환
    def get_target_rooms(self, df):
        min_room = df.loc[df['image_count'].idxmin()]['room']
        max_room = df.loc[df['image_count'].idxmax()]['room']
        mean_val = df['image_count'].mean()
        mean_room = df.iloc[(df['image_count'] - mean_val).abs().argsort()[:1]]['room'].values[0]
        return {'image_min': min_room, 'image_avg': mean_room, 'image_max': max_room}

    # 지정된 방 이미지 복사
    def copy_room_images(self, folder_room_map):
        for folder, room in folder_room_map.items():
            src = os.path.join(self.src_root, room, *self.image_subdir)
            dst = os.path.join(self.dst_root, folder)
            os.makedirs(dst, exist_ok=True)
            if os.path.exists(src):
                for fname in sorted(os.listdir(src), key=self.pixel_sort_key):
                    if fname.lower().endswith(self.image_extensions):
                        shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))

    # 샘플 인덱스 생성
    def get_sampling_indices(self, start, gap, total):
        return [min(start + i * gap, total - 1) for i in range(self.sample_image_count)]

    # 샘플링
    def sample_images(self):
        sampling_path = os.path.join(self.dst_root, 'sampling')
        os.makedirs(sampling_path, exist_ok=True)

        for folder in self.folder_names:
            image_folder = os.path.join(self.dst_root, folder)
            image_files = sorted([
                f for f in os.listdir(image_folder) if f.lower().endswith(self.image_extensions)
            ], key=self.pixel_sort_key)
            total_images = len(image_files)

            for strategy in self.strategies:
                gap = 1 if strategy == 'sequential' else max(1, math.floor(total_images / int(strategy.replace('gap', ''))))

                for set_idx in range(self.num_sets_per_strategy):
                    max_start = total_images - gap * (self.sample_image_count - 1)
                    if max_start <= 0:
                        continue
                    start_idx = random.randint(0, max_start)
                    indices = self.get_sampling_indices(start_idx, gap, total_images)
                    selected_files = [image_files[i] for i in indices]

                    save_dir = os.path.join(sampling_path, folder, strategy, f'set_{set_idx + 1}')
                    os.makedirs(save_dir, exist_ok=True)

                    for fname in selected_files:
                        src_path = os.path.join(image_folder, fname)
                        dst_path = os.path.join(save_dir, fname)
                        shutil.copy2(src_path, dst_path)

    ## 실행
    def run(self):
        # 이미지 수 기준 대상 방 선정
        df = self.get_room_image_counts()
        folder_room_map = self.get_target_rooms(df)

        # 대상 방 이미지 복사
        self.copy_room_images(folder_room_map)

        # 샘플링 수행
        self.sample_images()

        print("전체 작업 완료: 방 복사 및 샘플링")


# 실행
if __name__ == '__main__':
    sampler = ScannetSampler(
        src_root=r'E:\scannet\data_download\data',
        dst_root=r'C:\Users\최낙민.DESKTOP-RHND68Q\Desktop\scannet_test'
    )
    sampler.run()
