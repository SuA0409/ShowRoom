import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CLIP5ImageEvaluator:
    ## 경로 및 기본 설정
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

        ## 모델 로딩
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    # 이미지 경로를 받아 CLIP feature 추출
    def image_to_feature(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model.encode_image(image)
        return feature / feature.norm(dim=-1, keepdim=True)

    # 이미지 리스트에 대해 평균 코사인 유사도 계산
    def compute_avg_similarity(self, image_paths):
        features = torch.cat([self.image_to_feature(p) for p in image_paths], dim=0)
        sim_matrix = cosine_similarity(features.cpu().numpy())
        upper = sim_matrix[np.triu_indices(len(image_paths), k=1)]
        return upper.mean()

    # 다섯 장짜리 세트에 대해 평균 유사도/다양성 측정
    def evaluate_5_image_sets(self):
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

    ## 실행
    def run(self):
        print(" 5장 평균 유사도 측정 시작")
        df = self.evaluate_5_image_sets()
        df = df.sort_values(by=['folder', 'strategy', 'set'])
        df.to_csv(self.result_csv_path, index=False)
        print(" 저장 완료:", self.result_csv_path)
        return df


# 실행
if __name__ == '__main__':
    evaluator = CLIP5ImageEvaluator(
        base_path="/content/drive/MyDrive/sampling",           
        result_csv_path="/content/drive/MyDrive/sampling_5.csv"
    )
    evaluator.run()
