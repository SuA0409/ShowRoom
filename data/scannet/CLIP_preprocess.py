import os
import shutil
import pandas as pd
import random
import math

class ScannetSampler:
     """ScanNet 이미지 데이터를 기반으로 방 샘플링"""
     
    def __init__(self,
                 src_root,
                 dst_root,
                 image_subdir=None,
                 image_extensions=None,
                 sample_image_count=5,
                 num_sets_per_strategy=10,
                 strategies=None,
                 folder_names=None
                 ):
        """
        ScannetSampler 초기화 함수

        Args:
            src_root (str): 원본 ScanNet 데이터 경로
            dst_root (str): 복사 및 샘플링 결과 저장 경로
            image_subdir (list[str], optional): 이미지 하위 폴더 경로
            image_extensions (tuple, optional): 이미지 확장자 목록
            sample_image_count (int): 샘플당 이미지 수
            num_sets_per_strategy (int): 전략별 세트 수
            strategies (list[str], optional): 샘플링 전략 목록
            folder_names (list[str], optional): 저장 폴더 이름 목록
        """
        
        self.src_root = src_root                                                    
        self.dst_root = dst_root                                                    
        self.image_subdir = image_subdir or ['dslr', 'resized_undistorted_images']  
        self.image_extensions = image_extensions or ('.jpg', '.jpeg', '.png')       
        self.sample_image_count = sample_image_count                                
        self.num_sets_per_strategy = num_sets_per_strategy                          
        self.strategies = strategies or ['sequential', 'gap5', 'gap10', 'gap20', 'gap30', 'gap40']
        self.folder_names = folder_names or ['image_min', 'image_avg', 'image_max'] 
        self.pixel_sort_key = lambda x: int(''.join(filter(str.isdigit, x)))

    def get_room_image_counts(self):
        """
        각 방의 이미지 수를 계산하여 데이터프레임으로 반환

        Returns:
            pd.DataFrame: 방 이름과 이미지 개수를 포함한 데이터프레임
        """
        
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

    def get_target_rooms(self, df):
        """
        이미지 수 기준 최소, 평균, 최대 방을 선택

        Args:
            df (pd.DataFrame): 방 이름과 이미지 수를 포함한 데이터프레임

        Returns:
            dict: 'image_min', 'image_avg', 'image_max' 키에 해당하는 방 이름 딕셔너리
        """
        
        min_room = df.loc[df['image_count'].idxmin()]['room']
        max_room = df.loc[df['image_count'].idxmax()]['room']
        mean_val = df['image_count'].mean()
        mean_room = df.iloc[(df['image_count'] - mean_val).abs().argsort()[:1]]['room'].values[0]
        return {'image_min': min_room, 'image_avg': mean_room, 'image_max': max_room}

    def copy_room_images(self, folder_room_map):
        """
        선택된 방의 이미지를 지정된 폴더로 복사

        Args:
            folder_room_map (dict): 폴더 이름과 대응하는 방 이름 딕셔너리
        """
        
        for folder, room in folder_room_map.items():
            src = os.path.join(self.src_root, room, *self.image_subdir)
            dst = os.path.join(self.dst_root, folder)
            os.makedirs(dst, exist_ok=True)
            if os.path.exists(src):
                for fname in sorted(os.listdir(src), key=self.pixel_sort_key):
                    if fname.lower().endswith(self.image_extensions):
                        shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))

    def get_sampling_indices(self, start, gap, total):
        """
        시작 인덱스와 간격에 따라 이미지 인덱스 리스트를 생성

        Args:
            start (int): 시작 인덱스
            gap (int): 샘플링 간격
            total (int): 전체 이미지 수

        Returns:
            list[int]: 선택된 이미지 인덱스 목록
        """
        return [min(start + i * gap, total - 1) for i in range(self.sample_image_count)]

    def sample_images(self):
        """샘플링"""
        
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

    def run(self):
        """전체 파이프라인 실행 (방 선택, 이미지 복사, 샘플링 수행)"""
        
        # 이미지 수 기준 대상 방 선정
        df = self.get_room_image_counts()
        folder_room_map = self.get_target_rooms(df)

        # 대상 방 이미지 복사
        self.copy_room_images(folder_room_map)

        # 샘플링 수행
        self.sample_images()

        print("전체 작업 완료: 방 복사 및 샘플링")


if __name__ == '__main__':
    """메인 실행함수"""
    
    sampler = ScannetSampler(
        src_root=r'E:\scannet\data_download\data',
        dst_root=r'C:\Users\최낙민.DESKTOP-RHND68Q\Desktop\scannet_test'
    )
    sampler.run()
