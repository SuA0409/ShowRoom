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


## import
%cd s3r
import torch
import os
import time
import re
import json
import copy

from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

from show_loss import RKDLoss
from room2images_2 import batch_images_load

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# 가중치 로드
with open('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/configs/teacher_args.json', 'r') as f:
    teacher_args = json.load(f)
teacher_args['head_args']['conf_mode']=['exp', 1, float('inf')]
teacher_args['head_args']['depth_mode']= ['exp', float('-inf'), float('inf')]
student_args = copy.deepcopy(teacher_args)

## 수정 파라미터
# student_args['head_args']['layer_dims'] = [48, 96, 192, 384]
student_args['head_args']['feature_dim'] = 128


## Main Code
test_name = 'feature_dim128'

# initial path
student_model_save_path = f'/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/{test_name}'
rooms_path = '/content/drive/MyDrive/Scannet++/data_scannet_r_3'

# -- 1.  Load Data Path
rooms_name = [it for it in os.listdir('/content/drive/MyDrive/Scannet++') if it.endswith('.pt')][:9]
print(f'Number of Train Data : {len(rooms_name)*1000}')

# -- 2. Load Teahcer Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device : {device} \n')
 # Fast3r 제공 모델
torch.manual_seed(42) ; torch.cuda.manual_seed(42)
print('Building Teacher model ...')
teacher_model = Fast3R(**teacher_args)
teacher_model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/teacher.pth'))
teacher_model = teacher_model.to(device)
print('Finished Building Teacher model ! \n')

# -- 3. Build Students model
print('Building Student model ...')
student_model = Fast3R(**student_args)
student_model = student_model.to(device)
print('Finished Building Student model ! \n')

 # Sho3r 모델 저장 경로
os.makedirs(student_model_save_path, exist_ok=True)

 # load pre_model
if os.listdir(student_model_save_path):
    last_student_model = sorted([it for it in os.listdir(student_model_save_path)])[-1]
    student_model.load_state_dict(torch.load(student_model_save_path + f'/{last_student_model}'))
    print(f'loaded : {last_student_model} \n')
    start_epoch = int(re.findall(r'\d+', last_student_model)[0])
    start_i, start_batch = int(re.findall(r'\d+', last_student_model)[1]) // 4000, (int(re.findall(r'\d+', last_student_model)[1]) % 4000) // 4 + 1
    if start_i == len(rooms_name):
        start_i, start_batch = 0, 1
        start_epoch += 1
else:
    student_model.encoder = teacher_model.encoder
    student_model.decoder = teacher_model.decoder
    print("loaded : Teacher's en/decoder \n")
    start_epoch, start_i, start_batch = 1, 0, 1

# -- 4. Set Prams
epoch = 3

learning_rate = 1e-5
batch_size = 4
kd_loss = RKDLoss1()
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate
                            , weight_decay=0.05, betas=(0.9, 0.95))

# -- 5. Train
for e in range(start_epoch, epoch+1):
    print(f'########### START epoch : {e} ###########')

    teacher_model.eval()
    student_model.train()
    start_time = time.time()
    for i in range(start_i, (len(rooms_name))):

        print('Data Loading')
        load_time = time.time()
        data_all = torch.load(f'/content/drive/MyDrive/Scannet++/{rooms_name[i]}')
        print(f'{rooms_name[i]} Data Loading Time : {time.time() - load_time:6.3f}')

        for it in range(start_batch, 1001):
            # 데이터 가져오기
            batch_data = data_all.pop()

            optimizer.zero_grad()
            # prediction
            with torch.no_grad():
                teacher_pred = teacher_model(batch_data)
            student_pred = student_model(batch_data)

            # loss

            train_loss = kd_loss(student_pred, teacher_pred)
            train_loss.backward()
            optimizer.step()

            if it % 10 == 0:
                current_time =  time.time() - start_time
                info = f'epoch : {e}    batch : {it * batch_size + i * 4000:5d}    loss : {train_loss.item():.5f}    time : {current_time:6.3f}'
                print(info)
                with open(f'{student_model_save_path}/information_{test_name}.txt', 'a') as f:
                    f.write(info + '\n')
                start_time = time.time()

            if it % 500 == 0:
                print('\nSaving Model ...')
                torch.save(student_model.state_dict(), f'{student_model_save_path}/model_{e}_{(it * batch_size + i * 4000):05d}.pth')
                print('Saving Finished\n')



import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss1(nn.Module):
    def __init__(self, distance_weight=1, angle_weight=2):
        super(RKDLoss1, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    def pdist(self, e, eps=1e-8):
        # e = [b, c']
        e_square = e.pow(2).sum(dim=1) # b
        prod = e @ e.t() # [b, b]
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps) # [b, 1] + [1, b] - 2 * [b, b]
        # [b, b]

        res = res.sqrt() # [b, b]

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0 # 자신은 0
        return res

    def RKDDistance(self, student, teacher):
        # Input [b, c']
        with torch.no_grad():
            t_d = self.pdist(teacher)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


    def RKDAngle(self, student, teacher):
        """
        student, teacher: [N, D] 텐서, N: 배치 크기, D: 특징 차원
        각도 관계 손실 계산
        """
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1)) # [1, b, c'] - [b, 1, c'], [b, b, c']
            norm_td = F.normalize(td, p=2, dim=2) # [b, b, c']
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1) # [b*c'*b]

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

    def process_model_output(self, model_output):
        features = []
        # 모델 출력의 각 항목 처리
        for output_dict in model_output:
            # Global pointmap과 confidence 처리
            pointmap = output_dict['pts3d_in_other_view']  # [b, w, h, c]
            conf = output_dict['conf']  # [b, w, h]

            if len(pointmap.shape) == 4 and len(conf.shape) == 3:
                pointmap = pointmap.permute(0, 3, 2, 1)  # [b, c, h, w]
                conf = conf.permute(0, 2, 1).unsqueeze(1)  # [b, 1, h, w]

                # Confidence로 가중치 부여
                weighted_features = pointmap * conf  # [b, c, h, w]

                # Confidence의 합으로 나누어 정규화 (0으로 나누는 것 방지)
                conf_sum = conf.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                normalized_features = (weighted_features.sum(dim=(2, 3)) / conf_sum.squeeze(3).squeeze(2))  # [b, c]

                features.append(normalized_features)

        # 모든 특징을 연결
        if features:
            combined_features = torch.cat(features, dim=1)  # [b, s*c]
            return combined_features
        else:
            raise ValueError("특징 추출에 실패했습니다. 모델 출력 형식을 확인하세요.")

    def forward(self, student_output, teacher_output):
        student_features = self.process_model_output(student_output)  # [b, s*c]
        teacher_features = self.process_model_output(teacher_output)  # [b, s*c]

        # 거리 손실
        dist_loss = self.RKDDistance(student_features, teacher_features)

        # 각도 손실
        angle_loss = self.RKDAngle(student_features, teacher_features)

        # 가중치 적용한 최종 손실
        loss = self.distance_weight * dist_loss + self.angle_weight * angle_loss

        return loss
    


################ 시각화 ######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from fast3r.dust3r.utils.image import load_images

student_model = Fast3R(**student_args)
student_model = student_model.to(device)
student_model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/Half_Head_1/model_2_20000.pth'))

# teacher_model = Fast3R(**teacher_args)
# teacher_model = teacher_model.to(device)
# teacher_model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/teacher.pth'))

model = student_model
image_path = '/content/a'
images = load_images(image_path, size=512, verbose=True)

model.eval()
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
lit_module.eval()

output_dict, profiling_info = inference(
    images,
    model,
    device,
    dtype=torch.float32,
    verbose=True,
    profiling=True,
)

poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
    output_dict['preds'],
    niter_PnP=100,
    focal_length_estimation_method='first_view_from_global_head'
)


camera_poses = poses_c2w_batch[0]

for view_idx, pose in enumerate(camera_poses):
    print(f"Camera Pose for view {view_idx}:")
    print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix

for view_idx, pred in enumerate(output_dict['preds']):
    point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
    print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)  # shape: (b, 368, 512, 3), i.e., (1, Height, Width, XYZ)


# --- Align local point clouds to global space ---
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

