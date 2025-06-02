import cv2
import os
import numpy as np
import torch
import random
import concurrent.futures

def _load_process_image_cv2(img_path, size):
    img = cv2.imread(img_path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print(img_path)
    img = cv2.resize(img, (size, size * 3 // 4))  # 순서를 맞춰서 크기 조정
    img = img.astype(np.float32) / 255.0 * 2 - 1  # [-1, 1]로 정규화
    img = torch.from_numpy(img).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return img

def batch_images_load(rooms_path, rooms_name, batch_size, *, size=256, sample=3, device='cpu'):
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    rooms = list()
    imgs = [list() for _ in range(sample)]

    # Parallelize image loading using ThreadPoolExecutor or ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_img = {}

        # Start loading images concurrently
        for i in range(batch_size):
            room_path = os.path.join(rooms_path, rooms_name[i])
            imgs_name = [it for it in os.listdir(room_path) if any(it.lower().endswith(ext) for ext in supported_images_extensions)]

            # sellection number of sample
            imgs_name = random.sample(imgs_name, sample)
            for idx, img_name in enumerate(imgs_name):
                img_path = os.path.join(room_path, img_name)
                future = executor.submit(_load_process_image_cv2, img_path, size)
                future_to_img[future] = idx  # Track the future for ordering later

        # Collect results from futures
        for future in concurrent.futures.as_completed(future_to_img):
            idx = future_to_img[future]
            img = future.result()
            imgs[idx].append(dict(img=img, true_shape=torch.from_numpy(np.int32([size * 3 // 4, size])), idx=idx, instance=str(idx)))

    # Organize images into rooms
    for image in imgs:
        rooms.append({
            'img': torch.stack([d['img'] for d in image], dim=0),
            'true_shape': torch.stack([d['true_shape'] for d in image]).squeeze(),
            'idx': [d['idx'] for d in image],
            'instance': [d['instance'] for d in image],
        })

    # Move to device (GPU or CPU)
    for room in rooms:
        room['img'] = room['img'].to(device)
        room['true_shape'] = room['true_shape'].to(device)

    return rooms