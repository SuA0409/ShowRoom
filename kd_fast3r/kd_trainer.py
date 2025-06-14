import torch
import os
import time
import re
import yaml
import copy
import random

from kd_fast3r.utils.data_preprocess import batch_images_load
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from kd_fast3r.kd_loss import RKDLoss

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Fast3rTrainer:
    def __init__(self,
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
        self.test_name = test_name
        self.path = path
        self.arg_path = arg_path
        self.params = params

        # custom loss 사용
        self.kd_loss = RKDLoss()

        self._init_setting()

    def _init_setting(self):
        # 초기 주소 설정
        self.student_model_path = os.path.join(self.path['student_model_path'], self.test_name)

        # gpu 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device : {self.device} \n')

        # -- 1.  Load Data Path
        self.rooms_name = [it for it in os.listdir(self.path['rooms_path']) if it.endswith('.pt')]  # 이미지 processing cache
        print(f'Number of Train Data : {len(self.rooms_name) * 1000}')

    def _load_arguments(self,
                        student_args_path: str,
                        teacher_args_path: str = 'configs/teacher_args.yaml'
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

        return teacher_args, student_args

    def _load_teacher_model(self, teacher_args):
        torch.manual_seed(42); torch.cuda.manual_seed(42)

        # Fast3r 제공 모델
        print('Building Teacher model ...')
        self.teacher_model = Fast3R(**teacher_args)
        self.teacher_model.load_state_dict(torch.load(self.path['teacher_model_path']))
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        print('Finished Building Teacher model ! \n')

    def _load_student_model(self, student_args):
        # -- 3. Build Students model

        print('Building Student model ...')
        self.student_model = Fast3R(**student_args)

        try:
            # 가중치 폴더 로드
            student_weight_files = [it for it in os.listdir(self.student_model_path) if it.endswith('.pth')]
            # 가장 최신 가중치 파일 로드
            last_student_weight = sorted(student_weight_files, key=lambda x: len(x))[-1]
            self.checkpoint = torch.load(os.path.join(self.student_model_path, last_student_weight))

            # 모델 가중치만 로드
            self.student_model.load_state_dict(self.checkpoint['model_state_dict'])
            print(f'loaded : {last_student_weight} \n')

            start_epoch = int(re.findall(r'\d+', last_student_weight)[0])
            start_room_idx = int(re.findall(r'\d+', last_student_weight)[1]) // 1000

        except (FileNotFoundError, IndexError):
            # 모델 저장 경로
            os.makedirs(self.student_model_path, exist_ok=True)

            # teacher의 encoder와 decoder weight 가져오기
            self.student_model.encoder.load_state_dict(self.teacher_model.encoder.state_dict())
            self.student_model.decoder.load_state_dict(self.teacher_model.decoder.state_dict())
            print("loaded : Teacher's en/decoder \n")
            start_epoch, start_room_idx = 1, 0
            self.checkpoint = None

        os.makedirs(os.path.join(self.student_model_path, 'pred'), exist_ok=True)
        self.student_model = self.student_model.to(self.device)
        self.student_model.train()

        print('Finished Building Student model ! \n')

        return start_epoch, start_room_idx

    def make_kd_model(self):
        # arguments 불러오기
        teacher_args, student_args = self._load_arguments(**self.arg_path)

        # teacher/student_model 선언부
        self._load_teacher_model(teacher_args)
        start_epoch, start_room_idx = self._load_student_model(student_args)

        return start_epoch, start_room_idx

    def _load_check_point(self, start_epoch, start_room_idx):
        # optimzer와 scheduler 로드
        if self.checkpoint is None:
            # learning rate 0인 지점 스킵
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                # 첫 번째 lr은 0이므로 최초 step 실행
                self.scheduler.step()
        else:
            # optimizer와 scheduler 값 불러오기
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if 'scheduler_state_dict' in self.checkpoint:
                    self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

            # 에폭의 마지막까지 학습한 후 종료된 경우
            if start_room_idx == len(self.rooms_name):
                start_epoch += 1
                start_room_idx = 0
                # scheduler 전에 저장 했으므로, step 한 번 실행
                self.scheduler.step()

        self.optimizer.zero_grad()

        return start_epoch , start_room_idx

    def set_hyper_params(self, start_epoch, start_room_idx):
        # -- 4. Set Prams
        epochs = self.params['epochs']
        self.accum_iter = self.params.get('accum_iter', 1)
        learning_rate = float(self.params['learning_rate'])

        # AdamW 사용
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=self.params['optimizer']['weight_decay'],
            betas=tuple(self.params['optimizer']['betas'])
        )

        # scheduler를 사용
        if self.params['scheduler']['type'] == 'LinearWarmupCosineAnnealingLR':
            self.scheduler = LinearWarmupCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=self.params['scheduler']['warmup_epochs'],
                max_epochs=self.params['scheduler']['max_epochs'],
                eta_min=float(self.params['scheduler']['eta_min'])
            )

        start_epoch, start_room_idx = self._load_check_point(start_epoch, start_room_idx)

        return epochs, start_epoch, start_room_idx

    def data_loader(self, room_idx):
        data_all = torch.load(os.path.join(self.path['rooms_path'], self.rooms_name[room_idx]))
        random.shuffle(data_all)

        return data_all

    def train_model(self, batch_data):
        # device로 데이터 넘기기
        for batch_num, batch in enumerate(batch_data):
            batch_data[batch_num]['img'] = batch['img'].to(self.device)
            batch_data[batch_num]['true_shape'] = batch['true_shape'].to(self.device)

        # teacher & student prediction 시작
        with torch.no_grad():
            teacher_pred = self.teacher_model(batch_data)
        student_pred = self.student_model(batch_data)

        # loss 계산
        train_loss = self.kd_loss(student_pred, teacher_pred)
        train_loss = train_loss / self.accum_iter
        train_loss.backward()
        train_loss_value = train_loss.item()

        # colab issue
        torch.cuda.empty_cache()
        del teacher_pred, student_pred, train_loss, batch_data

        return train_loss_value

    def gradient_accumulation(self):
        # gradient accumulation으로 optimizer update
        self.optimizer.step()
        self.optimizer.zero_grad()

    def test(self, e):
        # test
        self.student_model.eval()
        test_output = dict()
        test_time = time.time()
        with torch.no_grad():
            for size in [256, 512]:
                test_data, _ = batch_images_load(rooms_path=self.path['test_image_path'], batch_size=1, size=512, sample=3)
                test_pred = self.student_model(test_data)
                test_output[str(size)] = test_pred

                try:
                    ## camera pose
                    poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
                        test_pred,
                        niter_PnP=100,
                        focal_length_estimation_method='first_view_from_global_head'
                    )

                    with open(f'{self.student_model_path}/pred/{size}.txt', 'a') as f:
                        f.write(f'{e:02d} : {poses_c2w_batch[0]} \n\n')
                except Exception as e:
                    print("Failed to save camera pose")

        torch.save(test_output, f'{self.student_model_path}/pred/{e:02d}.pth')

        self.student_model.train()

        print(f'Made test output ({time.time() - test_time:6.3f}) !')

        del test_output, test_data, test_pred, poses_c2w_batch