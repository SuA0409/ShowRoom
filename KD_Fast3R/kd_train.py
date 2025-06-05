import torch
import os
import time
import re
import yaml
import copy

from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.dust3r.utils.image import load_images

from KD_Fast3R.kd_loss import RKDLoss

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

def load_arguments(
        student_args_path: str,
        teacher_args_path: str='configs/teacher_args.yaml'
):
    '''teacher의 모델 구조를 가져오고, student의 바뀐 부분을 업데이트하여 student에도 저장
    Args:
        student_args_path (str): student의 argument가 저장된 yaml 파일 주소
        teacher_args_path (str): teacher의 argument가 저장된 yaml 파일 주소

    Returns:
        student_args (dict): model에 적용할 config가 들어있는 딕셔너리
        teacher_args (dict): model에 적용할 config가 들어있는 딕셔너리
    '''

    # 가중치 로드
     # teacher part
    with open(teacher_args_path, 'r', encoding='utf-8') as f:
        teacher_args = yaml.safe_load(f)
    teacher_args['head_args']['conf_mode'] = ['exp', 1, float('inf')]
    teacher_args['head_args']['depth_mode'] = ['exp', float('-inf'), float('inf')]
    student_args = copy.deepcopy(teacher_args)
     # student part
    with open(student_args_path, 'r', encoding='utf-8') as f:
        student_modify_args = yaml.safe_load(f)

    # 수정된 student의 args를 적용
    for k1, v1 in student_modify_args.items():
        for k2, v2 in v1.items():
            student_args[k1][k2] = v2

    return student_args, teacher_args


def train(
        test_name: str,
        path: dict,
        arg_path: dict,
        params: dict,
):
    ''' KD를 하기 위한 Main code
    Args:
        test_name (str): 테스트 이름을 지정
        path (dict):
            rooms_path (str): 방 데이터셋의 이름
            student_model_path (str): student 모델 저장 경로
            teacher_model_path (str): teacher 모델 가중치 경로
            test_image_path (str): 테스트 이미지 저장 경로
        arg_path (dict):
            student_args_path (str): student의 argument가 저장된 yaml 파일 주소
            teacher_args_path (str): teacher의 argument가 저장된 yaml 파일 주소
        params (dict): 기타 학습 파라미터
            epochs (int): 전체 학습 epoch 횟수
            learning_rate (float): 학습률
            accum_iter (int): Gradient Accumulation을 사용할 횟수
            optimizer (dict):
                type (str): optimizer 이름
                weight_decay (float): optimizer weight_decay에 대한 것 default=(0.05)
                betas (list): optimizer betas에 대한 것 default=(0.9, 0.95)
            scheduler (dict):
                type (str): scheduler 이름
                warmup_epochs (int): warmup에 사용할 epoch 수
                max_epochs (int): 전체 학습에 사용할 epoch 수
                eta_min (float): learning rate의 최소값
    '''

    # initial path
    student_model_path = os.path.join(path['student_model_path'], test_name)

    # gpu setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device} \n')

    # -- 1.  Load Data Path
    rooms_name = [it for it in os.listdir(path['rooms_path']) if it.endswith('.pt')]  # 전처리된 데이터셋
    print(f'Number of Train Data : {len(rooms_name) * 1000}')

    # -- 2. Load Teacher Model
    torch.manual_seed(42);
    torch.cuda.manual_seed(42)

    # load_arguments
    student_args, teacher_args = load_arguments(**arg_path)

    # Fast3r 제공 모델
    print('Building Teacher model ...')
    teacher_model = Fast3R(**teacher_args)
    teacher_model.load_state_dict(torch.load(path['teacher_model_path']))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    print('Finished Building Teacher model ! \n')

    # -- 3. Build Students model
    print('Building Student model ...')
    student_model = Fast3R(**student_args)

    try:
        # 가중치 폴더 로드
        student_weight_files = [it for it in os.listdir(student_model_path) if it.endswith('.pth')]
        # 가장 최신 가중치 파일 로드
        last_student_weight = sorted(student_weight_files, key=lambda x: len(x))[-1]
        checkpoint = torch.load(os.path.join(student_model_path, last_student_weight))

        # 모델 가중치만 로드
        student_model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded : {last_student_weight} \n')

        start_epoch = int(re.findall(r'\d+', last_student_weight)[0])
        start_room_idx = int(re.findall(r'\d+', last_student_weight)[1]) // 1000
    except (FileNotFoundError, IndexError):
        # 모델 저장 경로
        os.makedirs(student_model_path, exist_ok=True)

        # teacher의 encoder와 decoder weight 가져오기
        student_model.encoder.load_state_dict(teacher_model.encoder.state_dict())
        student_model.decoder.load_state_dict(teacher_model.decoder.state_dict())
        print("loaded : Teacher's en/decoder \n")
        start_epoch, start_room_idx = 1, 0
        checkpoint = None

    os.makedirs(os.path.join(student_model_path, 'pred'), exist_ok=True)
    student_model = student_model.to(device)
    student_model.eval()
    print('Finished Building Student model ! \n')

    # -- 4. Set Prams
    epochs = params['epochs']
    learning_rate = float(params['learning_rate'])
    accum_iter = params.get('accum_iter', 1)

    # AdamW 사용
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=params['optimizer']['weight_decay'],
        betas=tuple(params['optimizer']['betas'])
    )

    # scheduler를 사용
    if params['scheduler']['type'] == 'LinearWarmupCosineAnnealingLR':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=params['scheduler']['warmup_epochs'],
            max_epochs=params['scheduler']['max_epochs'],
            eta_min=float(params['scheduler']['eta_min'])
        )

    # custom loss 사용
    kd_loss = RKDLoss()

    # optimzer와 scheduler 로드
    if checkpoint is None:
        # learning rate 0인 지점 스킵
        try:
            scheduler.step()
        except NameError:
            print("Won't be using the scheduler.\n")
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except NameError:
            pass

        # 에폭의 마지막까지 학습한 후 종료된 경우
        if start_room_idx == len(rooms_name):
            start_epoch += 1
            start_room_idx = 0
            scheduler.step()

    del checkpoint

    # -- 5. Train
    for e in range(start_epoch, epochs + 1):
        print(f'########### START epoch : {e} ###########')
        student_model.train()
        start_time = time.time()

        for room_idx in range(start_room_idx, (len(rooms_name))):
            print('Data Loading')
            load_time = time.time()
            data_all = torch.load(os.path.join(path['rooms_path'], rooms_name[room_idx]))
            print(f'{rooms_name[room_idx]} Data Loading Time : {time.time() - load_time:6.3f}')

            optimizer.zero_grad()

            for it in range(1, len(data_all) + 1):
                # 데이터 가져오기
                batch_data = data_all.pop()

                # device로 데이터 넘기기
                for batch_num, batch in enumerate(batch_data):
                    batch_data[batch_num]['img'] = batch['img'].to(device)
                    batch_data[batch_num]['true_shape'] = batch['true_shape'].to(device)

                # teacher & student prediction
                with torch.no_grad():
                    teacher_pred = teacher_model(batch_data)
                student_pred = student_model(batch_data)

                # loss
                train_loss = kd_loss(student_pred, teacher_pred)
                train_loss = train_loss / accum_iter
                train_loss.backward()

                # gradient accumulation
                if it % accum_iter == 0 or it == len(data_all):
                    optimizer.step()
                    optimizer.zero_grad()

                if it % 10 == 0:
                    current_time = time.time() - start_time
                    info = f'''epoch : {e}    iter : {it + room_idx * 1000:4d}    loss : {train_loss.item():.5f}    lr : {optimizer.param_groups[0]['lr']:.4e}    time : {current_time:6.3f}'''
                    print(info)
                    with open(f'{student_model_path}/information_{test_name}.txt', 'a') as f:
                        f.write(info + '\n')
                    start_time = time.time()

                # colab issue
                torch.cuda.empty_cache()
                del teacher_pred, student_pred, train_loss, batch_data

            # 모델 저장
            if (room_idx + 1) == len(rooms_name) // 2 or (room_idx + 1) == len(rooms_name):
                print('\nSaving Model ...')
                torch.save({
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                    f'{student_model_path}/model_{e:02d}_{((room_idx + 1) * 1000):04d}.pth')
                print('Saving Finished\n')

            torch.cuda.empty_cache()
            del data_all

        # make test data
        student_model.eval()
        test_output = dict()
        test_time = time.time()
        with torch.no_grad():
            for size in [256, 512]:
                test_data = load_images(path['test_image_path'], size=size,
                                        verbose=False)
                test_pred = inference(test_data, student_model, device, dtype=torch.float32, verbose=False,
                                      profiling=False)
                test_output[str(size)] = test_pred

                ## camera pose
                poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
                    test_pred['preds'],
                    niter_PnP=100,
                    focal_length_estimation_method='first_view_from_global_head'
                )

                with open(f'{student_model_path}/pred/{size}.txt', 'a') as f:
                    f.write(f'{e:02d} : {poses_c2w_batch[0]} \n\n')

        print(f'Made test output ({time.time() - test_time:6.3f}) !')
        torch.save(test_output, f'{student_model_path}/pred/{e:02d}.pth')

        scheduler.step()
        start_room_idx = 0
        del test_output, test_data, test_pred, poses_c2w_batch, estimated_focals

    print('Finished training !')


def main(test_yaml_name='configs/test1.yaml'):
    with open(test_yaml_name, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train(**config)
