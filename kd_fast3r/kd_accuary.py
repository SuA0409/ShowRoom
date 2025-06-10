# 환경 설정
from google.colab import drive
drive.mount('/content/drive')
# 필요 라이브러리 설치
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List
import warnings
import os
import random

# 성능 평가 클래스
class Evaluator:
    def __init__(self, teacher_path, student_path):
        self.teacher_pointmap = self.load_teacher_pointmap_from_pt(teacher_path)
        self.student_pointmap = self.load_student_pointmap_from_pth(student_path)

    # teacher output를 { index : (x, y, z) } 형태로 변환
    def load_teacher_pointmap_from_pt(self, teacher_path: str, index: int = 0):
        teacher_data = torch.load(teacher_path, map_location=torch.device('cpu'))
        teacher_pointmap = dict()
        t_preds = teacher_data['preds']
        for i, v in enumerate(t_preds[0]['pts3d_in_other_view'].reshape(-1, 3)):
            teacher_pointmap[i] = v
        return teacher_pointmap

    # student output를 { index : (x, y, z) } 형태로 변환
    def load_student_pointmap_from_pth(self, student_path: str, index: int = 0):
        student_data = torch.load(student_path, map_location=torch.device('cpu'))
        student_pointmap = dict()
        s_preds = student_data['512']['preds']
        for i, v in enumerate(s_preds[0]['pts3d_in_other_view'].reshape(-1, 3)):
            student_pointmap[i] = v
        return student_pointmap

    # Chamber Distance : 전체 재구성 유사도 
    def chamber_distance(self, use_squared: bool = True, max_points=1000):
        # GPU 메모리 폭발 문제로, point 수를 제한하여 안정적으로 수행
        """
        CD(P,Q) = (1/|P|) * Σ min||p-q||² + (1/|Q|) * Σ min||q-p||²
        
        두 point cloud 간의 구조적 유사성을 측정
        각 점에서 가장 가까운 점까지의 거리를 양방향으로 계산.

        """
        def subsample_points(pointmap, max_points):
            keys = list(pointmap.keys())
            sampled_keys = random.sample(keys, min(len(keys), max_points))
            return np.array([pointmap[k] for k in sampled_keys])

        # 샘플링하여 point 수 제한
        teacher_points = subsample_points(self.teacher_pointmap, max_points)
        student_points = subsample_points(self.student_pointmap, max_points)

        # 거리 계산
        dist_gt_to_pred = cdist(teacher_points, student_points, metric='euclidean')
        min_dist_gt_to_pred = np.min(dist_gt_to_pred, axis=1)

        dist_pred_to_gt = cdist(student_points, teacher_points, metric='euclidean')
        min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)

        if use_squared:
            min_dist_gt_to_pred **= 2
            min_dist_pred_to_gt **= 2

        cd_gt_to_pred = np.mean(min_dist_gt_to_pred)
        cd_pred_to_gt = np.mean(min_dist_pred_to_gt)
        chamber_distance = cd_gt_to_pred + cd_pred_to_gt

        results = {
        'chamber_distance': float(chamber_distance),
        'cd_gt_to_pred': float(cd_gt_to_pred),
        'cd_pred_to_gt': float(cd_pred_to_gt),
        'num_gt_points': len(teacher_points),
        'num_pred_points': len(student_points),
        'distance_type': 'squared' if use_squared else 'euclidean'
        }

        return results

    # Point-wise L2 Distance : index의 정확도
    def pointwise_l2_distance(self):
        # 각 대응되는 포인트 간의 직접적인 거리를 측정
        
        # 공통 index만 사용 (두 pointmap에 모두 존재하는 index)
        common_indices = set(self.teacher_pointmap.keys()) & set(self.student_pointmap.keys())

        # 공통 index에 대해서만 좌표 추출
        gt_coords = np.array([self.teacher_pointmap[idx] for idx in sorted(common_indices)])
        pred_coords = np.array([self.student_pointmap[idx] for idx in sorted(common_indices)])

        # 각 포인트별 L2 거리 계산
        l2_distances = np.sqrt(np.sum((gt_coords - pred_coords) ** 2, axis=1))

        results = {
            'mean_l2_distance': float(np.mean(l2_distances)),
            'std_l2_distance': float(np.std(l2_distances)),
            'min_l2_distance': float(np.min(l2_distances)),
            'max_l2_distance': float(np.max(l2_distances)),
            'median_l2_distance': float(np.median(l2_distances))
        }

        return results

    # Per-axis MAE : 각 축 정확도
    def per_axis_mae(self):
        """
        MAE_x = (1/N) * Σ |xi - x̂i|

        X, Y, Z 축별 Mean Absolute Error 계산
        (어느 축에서 더 큰 오차가 발생하는지 파악 가능)
        
        """
        # 공통 index 찾기
        common_indices = set(self.teacher_pointmap.keys()) & set(self.student_pointmap.keys())

        # 좌표 추출 및 축별 분리
        gt_coords = np.array([self.teacher_pointmap[idx] for idx in sorted(common_indices)])
        pred_coords = np.array([self.student_pointmap[idx] for idx in sorted(common_indices)])

        # 각 축별 MAE 계산
        mae_x = np.mean(np.abs(gt_coords[:, 0] - pred_coords[:, 0]))
        mae_y = np.mean(np.abs(gt_coords[:, 1] - pred_coords[:, 1]))
        mae_z = np.mean(np.abs(gt_coords[:, 2] - pred_coords[:, 2]))

        # 전체 MAE (3D 공간에서의 평균 절대 오차)
        overall_mae = np.mean(np.abs(gt_coords - pred_coords))

        results = {
            'mae_x': float(mae_x),
            'mae_y': float(mae_y),
            'mae_z': float(mae_z),
            'overall_mae': float(overall_mae),
            'max_axis_mae': float(max(mae_x, mae_y, mae_z)),
            'min_axis_mae': float(min(mae_x, mae_y, mae_z)),
            'num_points': len(common_indices)
        }

        return results

    # Self-consistency : 동일한 장면을 서로 다른 뷰(view)로 입력했을 때, 복원된 결과가 서로 얼마나 일치하는가
    def self_consistency_check(self, neighbor_threshold: float = 0.1, max_points: int = 5000):
        # GPU 메모리 폭발 문제로, 사용할 포인트 수를 최대 5000개로 제한
        pred_coords = np.array(list(self.student_pointmap.values()))

        # 너무 많은 점 샘플링
        if len(pred_coords) > max_points:
            indices = np.random.choice(len(pred_coords), max_points, replace=False)
            pred_coords = pred_coords[indices]

        if len(pred_coords) < 2:
            return {'self_consistency_score': 1.0, 'num_outliers': 0}

        from scipy.spatial import KDTree
        tree = KDTree(pred_coords)
        neighbors_count = np.array([len(tree.query_ball_point(p, neighbor_threshold)) - 1 for p in pred_coords])

        mean_neighbors = np.mean(neighbors_count)
        std_neighbors = np.std(neighbors_count)
        outlier_threshold = max(1, mean_neighbors - 2 * std_neighbors)
        outliers = np.sum(neighbors_count < outlier_threshold)

        consistency_score = 1.0 - (outliers / len(pred_coords))

        results = {
            'self_consistency_score': float(consistency_score),
            'num_outliers': int(outliers),
            'mean_neighbors': float(mean_neighbors),
            'std_neighbors': float(std_neighbors),
            'total_points': len(pred_coords)
        }

        return results

    # SSIM : 밝기, 대비, 구조 등을 고려하여 얼마나 구조적/시각적으로 유사한가
    def compute_ssim_3d(self, grid_size: int = 32):
        # 3D point cloud를 3D grid로 voxelize한 후 SSIM을 계산
        def pointmap_to_voxel(pointmap, grid_size):
            # grid size : Voxel grid의 크기
            indices = np.array(list(pointmap.keys()))
            coords = np.array(list(pointmap.values()))

            # index 기준으로 정렬하여 일관성 확보
            sorted_indices = sorted(indices)
            coords = np.array([pointmap[idx] for idx in sorted_indices])

            if len(coords) == 0:
                return np.zeros((grid_size, grid_size, grid_size))

            # 좌표 정규화 (0~grid_size-1 범위로)
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            range_coords = max_coords - min_coords
            range_coords[range_coords == 0] = 1  # 0으로 나누기 방지

            normalized_coords = (coords - min_coords) / range_coords * (grid_size - 1)
            normalized_coords = np.round(normalized_coords).astype(int)

            # Voxel grid 생성
            voxel_grid = np.zeros((grid_size, grid_size, grid_size))
            for coord in normalized_coords:
                x, y, z = coord
                voxel_grid[x, y, z] = 1

            return voxel_grid

        try:
            gt_voxel = pointmap_to_voxel(self.teacher_pointmap, grid_size)
            pred_voxel = pointmap_to_voxel(self.student_pointmap, grid_size)

            # 3D SSIM 계산 (각 슬라이스별 SSIM의 평균)
            ssim_scores = []
            for i in range(grid_size):
                if np.sum(gt_voxel[i]) > 0 or np.sum(pred_voxel[i]) > 0:
                    score = ssim(gt_voxel[i], pred_voxel[i], data_range=1.0)
                    ssim_scores.append(score)

            return float(np.mean(ssim_scores)) if ssim_scores else 0.0

        except Exception as e:
            warnings.warn(f"SSIM 계산 중 오류: {e}")
            return 0.0

    def evaluate_all(self, teacher_output: Dict, student_output: Dict) -> Dict:
        results = {}

        print("-----  KD 성능 평가  -----")

        # 1. Chamber Distance
        results['chamber_distance'] = self.chamber_distance()
        print(">> Chamber Distance 평가 완료")
        
        # 2. Point-wise L2 Distance
        results['pointwise_l2'] = self.pointwise_l2_distance()
        print(">> Point-wise L2 Distance 평가 완료")
        
        # 3. Per-axis MAE
        results['per_axis_mae'] = self.per_axis_mae()
        print(">> Per-axis MAE 평가 완료")
        
        # 4. Self-consistency
        results['self_consistency'] = self.self_consistency_check()
        print(">> Self-consistency 평가 완료")
        
        # 5. 3D SSIM
        results['ssim_3d'] = self.compute_ssim_3d()
        print(">> 3D SSIM 평가 완료")
        
        print("-----  KD 평가 완료  -----")

        return results

    def print_summary(self, results: Dict):
        # 평가 결과 요약 출력
        print("\n\n" + "="*60)
        print("         Knowledge Distillation Evaluation Summary")
        print("="*60)
        
        # 1. Chamber Distance : 작을수록 구조적 차이 적음
        chamber = results['chamber_distance']
        print(f"\n1. Chamber Distance:")
        print(f"   • 총 Chamber Distance: {chamber['chamber_distance']:.6f}")  ## 비교 대상
        print(f"   • GT→Pred: {chamber['cd_gt_to_pred']:.6f}")
        print(f"   • Pred→GT: {chamber['cd_pred_to_gt']:.6f}")
        print(f"   • GT/Pred 포인트 수: {chamber['num_gt_points']} / {chamber['num_pred_points']}")

        # 2. Point-wise L2 Distance : 작을수록 포인트 간 위치 차이 적음
        l2 = results['pointwise_l2']
        print("\n2. Point-wise L2 Distance:")
        print(f"   • 평균 거리: {l2['mean_l2_distance']:.6f}")  ## 비교 대상
        print(f"   • 표준 편차: {l2['std_l2_distance']:.6f}")
        print(f"   • 최소 거리: {l2['min_l2_distance']:.6f}")
        print(f"   • 최대 거리: {l2['max_l2_distance']:.6f}")
        print(f"   • 중앙값 거리: {l2['median_l2_distance']:.6f}")

        # 3. Per-axis MAE : 작을수록 축별 평균 오차 적음
        mae = results['per_axis_mae']
        print(f"\n3. Per-axis MAE:")
        print(f"   • X축 MAE: {mae['mae_x']:.6f}")
        print(f"   • Y축 MAE: {mae['mae_y']:.6f}")
        print(f"   • Z축 MAE: {mae['mae_z']:.6f}")
        print(f"   • 전체 MAE: {mae['overall_mae']:.6f}")  ## 비교 대상
        print(f"   • 최대/최소 축 MAE: {mae['max_axis_mae']:.6f} / {mae['min_axis_mae']:.6f}")
        
        # 4. Self-consistency : 클수록(1에 가까움) 예측 결과 내 일관성 높음
        consistency = results['self_consistency']
        print(f"\n4. Self-consistency:")
        print(f"   • 일관성 점수: {consistency['self_consistency_score']:.4f}")  ## 비교 대상
        print(f"   • Outlier 수: {consistency['num_outliers']}")
        print(f"   • 평균 이웃 수: {consistency['mean_neighbors']:.2f}")
        
        # 5. 3D SSIM : 클수록 (최대 1) 구조 및 시각적 유사도 높음
        ssim_score = results['ssim_3d']
        print(f"\n5. 3D SSIM:")
        print(f"   • SSIM 점수: {ssim_score:.4f}")  ## 비교 대상
        
def accuary(teacher_path, student_path):
    # 평가 객체 생성 및 실행
    evaluator = Evaluator(teacher_path, student_path)
    results = evaluator.evaluate_all(evaluator.teacher_pointmap, evaluator.student_pointmap)
    evaluator.print_summary(results)

    return results

# 실행
if __name__ == "__main__":
    teacher_path = "/content/drive/MyDrive/INISW6_CV5/Fast3R/output data_for accuary/ex_teacher_output.pt"  # teahcer(Fast3R) 경로 (고정)
    student_path = "/content/drive/MyDrive/INISW6_CV5/Fast3R/output data_for accuary/18.pth"  # student(ShowRoom) 경로
    res_accuary = accuary(teacher_path, student_path)
