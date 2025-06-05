import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

## 경로 및 기본 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B/32"
BASE_PATH = "/content/drive/MyDrive/sampling"           #폴더 이름은 sampling
FOLDERS = ['images_min', 'images_avg', 'images_max']
STRATEGIES = ['sequential', 'gap5', 'gap10', 'gap20', 'gap30', 'gap40']
NUM_SETS = 10
RESULT_CSV_PATH = "/content/drive/MyDrive/sampling_5.csv"

## 모델 로딩
model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)

## 함수
# 이미지 경로를 받아 CLIP feature 추출
def image_to_feature(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = model.encode_image(image)
    return feature / feature.norm(dim=-1, keepdim=True)

# 이미지 리스트에 대해 평균 코사인 유사도 계산
def compute_avg_similarity(image_paths):
    features = torch.cat([image_to_feature(p) for p in image_paths], dim=0)
    sim_matrix = cosine_similarity(features.cpu().numpy())
    upper = sim_matrix[np.triu_indices(len(image_paths), k=1)]
    return upper.mean()

# 다섯 장짜리 세트에 대해 평균 유사도/다양성 측정
def evaluate_5_image_sets(base_path, folders, strategies, num_sets):
    results = []
    for folder in folders:
        for strategy in strategies:
            for set_idx in range(1, num_sets + 1):
                set_path = os.path.join(base_path, folder, strategy, f'set_{set_idx}')
                if not os.path.exists(set_path):
                    continue

                image_files = sorted(
                    [os.path.join(set_path, f) for f in os.listdir(set_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                )

                if len(image_files) != 5:
                    continue

                try:
                    avg_sim = compute_avg_similarity(image_files)
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
def run_clip_5_image_evaluation():
    print(" 5장 평균 유사도 측정 시작")
    df = evaluate_5_image_sets(BASE_PATH, FOLDERS, STRATEGIES, NUM_SETS)
    df = df.sort_values(by=['folder', 'strategy', 'set'])
    df.to_csv(RESULT_CSV_PATH, index=False)
    print(" 저장 완료:", RESULT_CSV_PATH)
    return df

# 실행
if __name__ == '__main__':
    run_clip_5_image_evaluation()
