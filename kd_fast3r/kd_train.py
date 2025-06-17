import yaml
import time
import torch

from kd_fast3r.kd_trainer import Fast3rTrainer

test_yaml_name = 'configs/test4.yaml'  # test_yaml_name 파일 주소

# yaml 파일을 불러와 config에 저장
with open(test_yaml_name, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

showroom = Fast3rTrainer(**config)

start_epoch, start_room_idx = showroom.make_kd_model()  # teacher와 student를 생성

epochs, start_epoch, start_room_idx = showroom.set_hyper_params(start_epoch, start_room_idx)  # hyper parameters를 선언

for e in range(start_epoch, epochs + 1):
    print(f'########### START epoch : {e} ###########')

    for room_idx in range(start_room_idx, (len(showroom.rooms_name))):
        print('Data Loading')
        load_time = time.time()
        data_all = showroom.data_loader(room_idx)
        print(f'{showroom.rooms_name[room_idx]} Data Loading Time : {time.time() - load_time:6.3f}')

        start_time = time.time()

        for it in range(1, len(data_all) + 1):
            # 데이터 가져오기 pop을 하여 colab issue 해결
            batch_data = data_all.pop()

            loss = showroom.train_model(batch_data)

            if it % showroom.accum_iter == 0:
                showroom.gradient_accumulation()

            if it % 10 == 0:
                info = (f"epoch : {e}    iter : {it + room_idx * 1000:5d}    loss : {loss:.5f}    "
                        f"lr : {showroom.optimizer.param_groups[0]['lr']:.4e}    time : {time.time() - start_time:6.3f}")
                print(info)
                with open(f'{showroom.student_model_path}/information_{showroom.test_name}.txt', 'a') as f:
                    f.write(info + '\n')
                start_time = time.time()

        # 모델 저장; 용량 7GB이므로 주의 필요
        print('\nSaving Model ...')
        ckpt = {
            'model_state_dict': showroom.student_model.state_dict(),
            'optimizer_state_dict': showroom.optimizer.state_dict(),
        }
        if hasattr(showroom, 'scheduler'):
            ckpt['scheduler_state_dict'] = showroom.scheduler.state_dict()

        torch.save(ckpt,
                   f'{showroom.student_model_path}/model_{e:02d}_{((room_idx + 1) * 1000):05d}.pth')
        print('Saving Finished\n')

        torch.cuda.empty_cache()
        del data_all

    showroom.test(e)

    showroom.scheduler.step()
    start_room_idx = 0

print('Finished training!')
