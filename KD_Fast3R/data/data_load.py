# Google mount
from google.colab import drive
drive.mount('/content/drive')


# 경로 설정
rooms_path = '/content/drive/MyDrive/Scannet++/data_scannet_r_3'
dataset_path = '/content/drive/MyDrive/Scannet++'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num = 52001

dataset = list()
idx = 0
for i in range(num, num+4000, 4):
    rooms_name = [str(j) for j in range(i, i+4)]
    dataset.append(batch_images_load(rooms_path, rooms_name ,4, sample=5, device=device))
    if (i-1) % 400 == 0:
        print(int((idx/10)*100), '%'); idx +=1
torch.save(dataset, dataset_path+f'/data_torch-{num+4000-1}.pt')


# 경로 이름
import os
os.listdir(f'{rooms_path}/5002')

import time
s = time.time()
a = torch.load(dataset_path+f'/data_torch-{num+4000-1}.pt')
print(time.time()-s)
len(a)


# Main
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
            'img': torch.stack([d['img'] for d in image]),
            'true_shape': torch.stack([d['true_shape'] for d in image]).squeeze(),
            'idx': [d['idx'] for d in image],
            'instance': [d['instance'] for d in image],
        })
    for room in rooms:
        room['img'].to(device)
        room['true_shape'].to(device)
    return rooms