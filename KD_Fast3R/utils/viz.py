## 사전 설치
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Colab Notebooks/Model/ShowRoom
!pip install -r requirements.txt

!pip install --upgrade --no-cache-dir --force-reinstall numpy==2.2.5
import os
os.kill(os.getpid(), 9)


# load fast3r git clone
%cd /content/drive/MyDrive/Colab Notebooks/Model/ShowRoom
!pip install -e .


## --1
%cd s3r
import torch
import json
import copy

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/configs/teacher_args.json', 'r') as f:
    teacher_args = json.load(f)
teacher_args['head_args']['conf_mode']=['exp', 1, float('inf')]
teacher_args['head_args']['depth_mode']= ['exp', float('-inf'), float('inf')]
student_args = copy.deepcopy(teacher_args)

## 수정 파라미터
student_args['head_args']['layer_dims'] = [48, 96, 192, 384]
# student_args['head_args']['feature_dim'] = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --2
import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3raa.fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# --- Setup ---
who = True   # True면, student_model, False면, teacher_model
model_name = '10_12000'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = '/content/drive/MyDrive/Colab Notebooks/Model/test_view'
# image_path = '/content/drive/MyDrive/Scannet++/data_scannet_r_3/10'
# image_path = '/content/drive/MyDrive/Scannet++/test'

if who:
    checkpoint = torch.load(f'/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/layer_dims_half/model_{model_name}.pth')
    student_model = Fast3R(**student_args)
    student_model = student_model.to(device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    model = student_model.to(device)
else:
    teacher_model = Fast3R(**teacher_args)
    teacher_model = teacher_model.to(device)
    teacher_model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/teacher.pth'))
    model = teacher_model.to(device)

images = load_images(image_path, size=512, verbose=False)
output_dict = inference(images, model, device, dtype=torch.float32, verbose=False, profiling=False)
# Create a lightweight lightning module wrapper for the model.
# This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

model.eval()
lit_module.eval()

# --- Estimate Camera Poses ---
# This step estimates the camera-to-world (c2w) poses for each view using PnP.
poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
    output_dict['preds'],
    niter_PnP=100,
    focal_length_estimation_method='first_view_from_global_head'
)
# poses_c2w_batch is a list; the first element contains the estimated poses for each view.
camera_poses = poses_c2w_batch[0]

# Print camera poses for all views.
for view_idx, pose in enumerate(camera_poses):
    print(f"Camera Pose for view {view_idx}:")
    print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix

for view_idx, pred in enumerate(output_dict['preds']):
    point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
    print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)

# --3
lit_module.align_local_pts3d_to_global(
    preds=output_dict['preds'],
    views=output_dict['views'],
    min_conf_thr_percentile=85
)

from fast3r.viz.viser_visualizer import start_visualization

server = start_visualization(
    output=output_dict,
    min_conf_thr_percentile=10,
    global_conf_thr_value_to_drop_view=1.5,
    point_size=0.0004,
)

print("🌐 3D 시각화 링크:", server.request_share_url())
