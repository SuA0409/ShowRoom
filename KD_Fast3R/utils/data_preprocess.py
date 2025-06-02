import torch
import os
from KD_Fast3R.utils.data_process import batch_images_load

# data loader의 시간적 한계로 사전에 전처리 후 사용 <- 사전 전처리 코드
def data_preprocesser(rooms_path='/content/drive/MyDrive/Scannet++/data_scannet_fin', dataset_path='/content/drive/MyDrive/Scannet++/preprocessv2'):
    for num in range(0, 38):
        room_path = os.path.join(rooms_path, f'data_scannet_r_{num}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = [None] * 1000
        room_files = sorted(os.listdir(room_path))

        # 40gb VRAM에서 사용 가능한 batch size는 4로 고정
        for i in range(1, 4001, 4):
            rooms_name = [room_files.pop() for _ in range(4)]
            dataset[i//4] = batch_images_load(room_path, rooms_name, 4, size=256, sample=5, device=device)
        torch.save(dataset, dataset_path+f'/data_torch-{num}.pt')

        del rooms_name, dataset, room_path, dataset_path, device
        print(f'data_torch-{num}.pt 저장 완료')