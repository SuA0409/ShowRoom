import cv2
import os
import numpy as np
import torch
import random
import concurrent.futures


# 이미지를 torch로 바꾸는 함수
def _load_image_process(img_data, size=256):
    if isinstance(img_data, str):
        img = cv2.imread(img_data)
    else:
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    # 이미지 샘플링에서의 오류를 검출
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size * 3 // 4))  # 이미지를 256, 192로 resize한 상태로 받은 후 다시 한 번 점검

        img_color = img.astype(np.float32) / 255.0  # [0, 1]로 정규화
        img_tensor = img.astype(np.float32) / 255.0 * 2 - 1  # [-1, 1]로 정규화
        img_tensor = torch.from_numpy(img_tensor).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    except:
        print(img_data)
        return

    return img_tensor, img_color


def batch_images_load(rooms_path,
                      rooms_name=None,
                      batch_size=4,
                      *,
                      size=256,
                      sample=5,
                      ):
    '''한 torch file에 4_000개의 room과, 각각 5장의 iamges 존재
        각 방을 batch_size(b) 각 방안에 이미지를 sample(s)로 표시
        preprocesser를 위한 thread를 사용하는 전용 함수
    Args:
        rooms_path (str): 방의 모음이 들어 있는 path
        rooms_name (list): 각각 방의 모음의 이름
        batch_size (int): 학습 시 batch_size
        size (int): 이미지의 가장 변의 길이; 기댓값은 width
        sample (int): 각각의 방의 이미지의 선택 계수
    Return:
        rooms (list): 각각의 sample의 크기만큼 dictionary가 존재
            (dict):
            img (torch.tensor): normalization된 batch_size 이미지의 데이터, [s, b, c, h, w]의 shape
            true_shape (torch.tensor): 이미지의 크기를 가지고 있는 텐서 [s, b, 2]의 shape
            idx (list[int]): 이미지의 index를 표시 ; Fast3r에서는 사용하였지만, 본 학습에서는 사용 X [s, b]
            instance (list[str]): 이미지의 instance를 표시; Fast3r에서는 사용하였지만, 본 학습에서는 사용 X [s, b]
    '''

    supported_images_extensions = [".jpg", ".jpeg", ".png"]  # 이미지의 확장자를 제한
    rooms = list()
    imgs = [list() for _ in range(sample)]  # 샘플 개수만큼 미리 생성
    colors = [list() for _ in range(sample)]

    if rooms_name is None:
        assert batch_size == 1, 'batch size must be 1'

    # 전처리 속도 향상을 위해 thread를 사용
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_img = {}

        # batch_size 만큼 전처리 실행
        for i in range(batch_size):
            # room_path
            room_path = rooms_path if rooms_name is None else os.path.join(rooms_path, rooms_name[i])

            imgs_name = [it for it in os.listdir(room_path) if
                         any(it.lower().endswith(ext) for ext in supported_images_extensions)]

            # sample size 만큼 이미지를 sampling한 후 가져옴
            imgs_name = random.sample(imgs_name, sample)
            for idx, img_name in enumerate(imgs_name):
                img_path = os.path.join(room_path, img_name)
                future = executor.submit(_load_image_process, img_path, size)
                future_to_img[future] = idx

        # 전처리 완료후 저장
        for future in concurrent.futures.as_completed(future_to_img):
            idx = future_to_img[future]
            img, color = future.result()
            imgs[idx].append(
                dict(img=img, true_shape=torch.from_numpy(np.int32([size * 3 // 4, size])), idx=idx, instance=str(idx)))
            colors[idx].append(color)

    # Fast3R 모델에서 원하는 데이터 타입으로 형태를 변경
    for image in imgs:
        rooms.append({
            'img': torch.stack([d['img'] for d in image], dim=0),
            'true_shape': torch.stack([d['true_shape'] for d in image]),
            'idx': [d['idx'] for d in image],
            'instance': [d['instance'] for d in image],
        })

    return rooms, colors


def server_images_load(files, size=512):
    sample = len(files)
    imgs = [list() for _ in range(sample)]
    colors = [list() for _ in range(sample)]
    rooms = list()

    for idx, f in enumerate(files.values()):
        image_bytes = f.read()
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        img, color = _load_image_process(file_bytes, size=size)
        imgs[idx].append(
                dict(img=img, true_shape=torch.from_numpy(np.int32([size * 3 // 4, size])), idx=idx, instance=str(idx)))
        colors[idx].append(color)

    for image in imgs:
        rooms.append({
            'img': torch.stack([d['img'] for d in image], dim=0),
            'true_shape': torch.stack([d['true_shape'] for d in image]),
            'idx': [d['idx'] for d in image],
            'instance': [d['instance'] for d in image],
        })

    return rooms, colors