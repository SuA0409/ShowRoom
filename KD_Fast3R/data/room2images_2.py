from PIL.ImageOps import exif_transpose
import PIL.Image
import torchvision.transforms as tvf
import numpy as np
import os
import torch
import random

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 제일 긴 변을 고정해서 사이즈를 조정
def _resize_pil_image(img, long_edge_size):
    S = max(img.size)

    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC

    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize((long_edge_size, long_edge_size * 3 // 4), interp)

def batch_images_load(rooms_path, rooms_name, batch_size, size=256, *,sample=3 ,target_ratio=4/3 ,device='cpu'):
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    rooms = list()
    imgs = [list() for _ in range(sample)]

    for i in range(batch_size):
        room_path = os.path.join(rooms_path, rooms_name[i])
        imgs_name = [it for it in os.listdir(room_path) if any(it.lower().endswith(ext) for ext in supported_images_extensions)]

        imgs_name = random.sample(imgs_name, sample)
        for idx, img_name in enumerate(imgs_name):
            img_path = os.path.join(room_path, img_name)
            img = exif_transpose(PIL.Image.open(img_path)).convert("RGB")
            img = _resize_pil_image(img, size)

            imgs[idx].append(dict(img=ImgNorm(img), true_shape=torch.from_numpy(np.int32([img.size[::-1]])), idx=idx, instance=str(idx)))

    for image in imgs:
        rooms.append({
            'img': torch.stack([d['img'] for d in image]).to(device),
            'true_shape': torch.stack([d['true_shape'] for d in image]).squeeze().to(device),
            'idx': [d['idx'] for d in image],
            'instance': [d['instance'] for d in image],
        })
    return rooms
