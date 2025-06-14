import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CLIP5ImageEvaluator:
     """CLIP 모델을 활용하여 이미지 5장 세트에 대한 평균 유사도 및 다양성을 평가하는 클래스"""
     
    def __init__(
                 self,
                 base_path,
                 result_csv_path,
                 folders=None,
                 strategies=None,
                 num_sets=10,
                 model_name="ViT-B/32"
                 ):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_path = base_path
        self.result_csv_path = result_csv_path
        self.folders = folders or ['images_min', 'images_avg', 'images_max']
        self.strategies = strategies or ['sequential', 'gap5', 'gap10', 'gap20', 'gap30', 'gap40']
        self.num_sets = num_sets
        self.model_name = model_name

        # CLIP 모델 로드
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def image_to_feature(self, image_path):
        """
        이미지 경로를 받아 CLIP 이미지 임베딩(feature)을 추출

        Args:
            image_path (str): 이미지 파일 경로

        Returns:
            torch.Tensor: 정규화된 CLIP 이미지 feature 벡터 (1, dim)
        """
        
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model.encode_image(image)
        return feature / feature.norm(dim=-1, keepdim=True)

    def compute_avg_similarity(self, image_paths):
        """
        이미지 리스트를 입력 받아 pairwise 코사인 유사도 계산

        Args:
            image_paths (list[str]): 5장 이미지 경로 리스트

        Returns:
            float: 평균 코사인 유사도
        """
        
        features = torch.cat([self.image_to_feature(p) for p in image_paths], dim=0)
        sim_matrix = cosine_similarity(features.cpu().numpy())
        upper = sim_matrix[np.triu_indices(len(image_paths), k=1)]
        return upper.mean()

    def evaluate_5_image_sets(self):
        """
        설정된 폴더 및 샘플링 전략에 따라 각 5장 이미지 세트의 평균 유사도 및 다양성 계산

        Returns:
            pd.DataFrame: 평가 결과 테이블 (컬럼: folder, strategy, set, avg_similarity, diversity)
        """
        results = []
        for folder in self.folders:
            for strategy in self.strategies:
                for set_idx in range(1, self.num_sets + 1):
                    set_path = os.path.join(self.base_path, folder, strategy, f'set_{set_idx}')
                    if not os.path.exists(set_path):
                        continue

                    image_files = sorted(
                        [os.path.join(set_path, f) for f in os.listdir(set_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    )

                    if len(image_files) != 5:
                        continue

                    try:
                        avg_sim = self.compute_avg_similarity(image_files)
                        results.append({
                            'folder': folder,
                            'strategy': strategy,
                            'set': f'set_{set_idx}',
                            'avg_similarity': avg_sim,
                            'diversity': 1 - avg_sim
                        })
                    except Exception as e:
                        print(f"Error in {folder}/{strategy}/{set_idx}: {e}")
        return pd.DataFrame(results)

    def run(self):
        """
        전체 평가를 실행

        Returns:
            pd.DataFrame: 정렬된 결과 테이블
        """
        print(" 5장 평균 유사도 측정 시작")
        df = self.evaluate_5_image_sets()
        df = df.sort_values(by=['folder', 'strategy', 'set'])
        df.to_csv(self.result_csv_path, index=False)
        print(" 저장 완료:", self.result_csv_path)
        return df


if __name__ == '__main__':
    """메인 실행부: CLIP5ImageEvaluator를 실행하여 결과를 평가 및 저장"""
    
    evaluator = CLIP5ImageEvaluator(
        base_path="/content/drive/MyDrive/sampling",           
        result_csv_path="/content/drive/MyDrive/sampling_5.csv"
    )
    evaluator.run()
