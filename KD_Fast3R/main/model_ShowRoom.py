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
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# 가중치 로드
with open('/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/configs/teacher_args.json', 'r') as f:
    teacher_args = json.load(f)
teacher_args['head_args']['conf_mode']=['exp', 1, float('inf')]
teacher_args['head_args']['depth_mode']= ['exp', float('-inf'), float('inf')]
student_args = copy.deepcopy(teacher_args)

## 수정 파라미터
student_args['head_args']['layer_dims'] = [48, 96, 192, 384]
# student_args['head_args']['feature_dim'] = 128


## Main Code
test_name = 'layer_dims_half'

# initial path
student_model_save_path = f'/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/.model/{test_name}'
rooms_path = '/content/drive/MyDrive/Scannet++/data_scannet_r_3'

# -- 1.  Load Data Path
rooms_name = [it for it in os.listdir('/content/drive/MyDrive/Scannet++') if it.endswith('.pt')]
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
student_model_path = [it for it in os.listdir(student_model_save_path) if it.endswith('.pth')]
if student_model_path:
    last_student_model = sorted(student_model_path)[-1]
    checkpoint = torch.load(student_model_save_path + f'/{last_student_model}')

    student_model.load_state_dict(checkpoint['model_state_dict'])
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
    checkpoint = None

student_model.eval()

# -- 4. Set Prams
warmup_epochs = 1
epoch = 50   # epoch : 전체 학습(epoch) 기간
learning_rate = 1e-5
eta_min = 1e-6   # learning_rate = 학습률의 최솟값
batch_size = 4
kd_loss = RKDLoss1()
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate
                            , weight_decay=0.05, betas=(0.9, 0.95))
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=epoch, eta_min=eta_min)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
del checkpoint


# -- 5. Train
for e in range(start_epoch, epoch+1):
    print(f'########### START epoch : {e} ###########')
    scheduler.step()
    teacher_model.eval()
    student_model.train()
    start_time = time.time()
    for i in range(start_i, (len(rooms_name))):
        print('Data Loading')
        load_time = time.time()
        data_all = torch.load(f'{rooms_path}/{rooms_name[i]}')
        print(f'{rooms_name[i]} Data Loading Time : {time.time() - load_time:6.3f}')

        for it in range(start_batch, 1001):
            # 데이터 가져오기
            batch_data = data_all[it-1]

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
                info = f'''epoch : {e}    batch : {it * batch_size + i * 4000:5d}    loss : {train_loss.item():.5f}    lr : {optimizer.param_groups[0]['lr']:.4e}    time : {current_time:6.3f}'''
                print(info)
                with open(f'{student_model_save_path}/information_{test_name}.txt', 'a') as f:
                    f.write(info + '\n')
                start_time = time.time()

            if it % 1000 == 0:
                print('\nSaving Model ...')
                torch.save({
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                    f'{student_model_save_path}/model_{e}_{(it * batch_size + i * 4000):05d}.pth')
                print('Saving Finished\n')

            del teacher_pred, student_pred, train_loss
        start_batch = 1
    start_i = 0
