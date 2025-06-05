## pt와 pose를 뽑기 위한 Fast3R의 demo 코드
import os
import torch
import numpy as np
import cv2
import concurrent.futures
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

## 이미지 불러오기 및 전처리
def load_and_preprocess_images(image_dir, size=512):
    supported_exts = [".jpg", ".jpeg", ".png"]
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in supported_exts
    ])

    def _load_process_image(img_path):
        img = cv2.imread(img_path)                        # BGR로 읽힘
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # RGB로 변환
        img = cv2.resize(img, (size, size * 3 // 4))      # 4:3 비율 유지
        img = img.astype(np.float32) / 255.0 * 2 - 1      # [-1, 1] 정규화
        return torch.from_numpy(img).float().permute(2, 0, 1)  # (C, H, W)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        imgs = list(executor.map(_load_process_image, image_paths))

    return imgs

## Fast3r 입력 형태로 전처리
def prepare_fast3r_input(imgs, device):
    images = []
    for idx, img in enumerate(imgs):
        C, H, W = img.shape
        images.append({
            "img": img.unsqueeze(0).to(device),              # (1, C, H, W)
            "true_shape": torch.tensor([[H, W]]).to(device), # (1, 2)
            "idx": idx,
            "instance": str(idx)
        })
    return images

## Point cloud 저장(.npz)
def save_point_clouds(preds, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    npz_dict = {}
    for i, pred in enumerate(preds):
        pts = pred["pts3d_global"].cpu().numpy().astype(np.float32).reshape(-1, 3)
        npz_dict[f"pts{i}"] = pts
    np.savez(save_path, **npz_dict)
    print(f" Saved combined point clouds to {save_path}")

# 카메라 포즈 추정
def save_camera_poses(preds, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Fast3R의 MultiView 모듈을 이용한 camera pose 추정
    poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
        preds,
        niter_PnP=100,
        focal_length_estimation_method="first_view_from_global_head"
    )
    camera_poses = poses_c2w_batch[0]  # List of (4, 4) numpy arrays

    
    # --- 문자열로 변환하여 저장
    with open(save_path, "w") as f:
        f.write("[\n")
        for pose in camera_poses:
            pose_str = np.array2string(pose, separator=', ', precision=8, suppress_small=False)
            f.write(f" np.array({pose_str}, dtype=float32),\n")
        f.write("]")
    print(f" Saved all camera poses to {save_path}")

## 전체 실행 함수
def main(
    image_dir,
    point_save_path,
    pose_save_path,
    model_name="jedyang97/Fast3R_ViT_Large_512",
    image_size=512,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Fast3R.from_pretrained(model_name).to(device)
    model.eval()

    # # Load model - teacher & student
    # model = Fast3R(model_name).to(device)
    # model.eval()

    # Load & process images
    imgs = load_and_preprocess_images(image_dir, size=image_size)
    inputs = prepare_fast3r_input(imgs, device)

    # Inference
    with torch.no_grad():
        preds = model(inputs)

    # 결과 저장
    save_point_clouds(preds, point_save_path)
    save_camera_poses(preds, pose_save_path)

## 실행
if __name__ == "__main__":
    # === 설정 ===
    image_dir = "/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Images"
    point_save_path = "/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Pts/views.npz"
    pose_save_path = "/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Pose/pose.txt"
    model_name = "jedyang97/Fast3R_ViT_Large_512"

    main(image_dir, point_save_path, pose_save_path, model_name=model_name)
