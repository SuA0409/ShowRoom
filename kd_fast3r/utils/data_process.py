import torch
import os
from kd_fast3r.utils.data_preprocess import batch_images_load

def data_processer(
    rooms_path: str='/content/drive/MyDrive/Scannet++/data_scannet_fin', 
    dataset_path: str='/content/drive/MyDrive/Scannet++/preprocessv2'
):
    '''이미지 batch크기만큼 전처리 후 torch로 저장 <- dataloader의 학습속도가 매우 느림 ; 사전 전처리의 필요성
    Args:
        rooms_path (str): 방의 모음이 들어 있는 path
        dataset_path (str): 저장된 파일을 저장할 path
    '''
    for num in range(0, 38): # 총 38개의 데이터 모음을 생성
        room_path = os.path.join(rooms_path, f'data_scannet_r_{num}')

        # dataset을 None으로 미리 선언 <- colab에서의 저장 오류 해결
        dataset = [None] * 1000
        room_files = sorted(os.listdir(room_path))
        
        # 40gb VRAM(A100)에서 사용 가능한 최대 효율 batch size는 4로 고정
        for i in range(1, 4001, 4):
            rooms_name = [room_files.pop() for _ in range(4)] # colab의 ram explosion을 방지하기 위해 pop으로 코딩
            rooms, _ = batch_images_load(room_path, rooms_name, 4, size=256, sample=5)
            dataset[i//4] = rooms
            
        torch.save(dataset, dataset_path+f'/data_torch-{num}.pt') # pt파일로 데이터 저장 후 load ; 병목 현상 최소화를 목적

        del rooms_name, dataset, room_path, dataset_path # colab issue
        
        print(f'data_torch-{num}.pt 저장 완료')
