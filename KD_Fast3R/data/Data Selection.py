# Import
!pip install matplotlib pillow

import os, time
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output
import shutil
import warnings; warnings.filterwarnings('ignore', category=UserWarning)


# 이미지 로드
def load_rooms_number(root):
    room_list = list()
    for name in ['del.txt', 'save.txt']:
        try:
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            room_list.extend(lines)
        except:
            pass
    return set(room_list)

def save_rooms_number(save_root, file_name, room_number):
    with open(f'{save_root}/{file_name}.txt', 'a', encoding='utf-8') as file:
        file.write(f'{room_number}\n')

def show_select_images(imgs_path, rn):
    imgs = os.listdir(imgs_path)

    ## show images

    _, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.ravel()
    imgs.sort()
    print(f'방 번호 : {rn}')

    for i, img_name in enumerate(imgs):
        img_path = os.path.join(imgs_path, img_name)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_name)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    time.sleep(0.1)
    clear_output(wait=True)

    ## select imagess

    user_input = map(int, input("이미지 번호 입력: ").split())
    user_input = list(set(filter(lambda n: 0 <= n < len(imgs), user_input)))

    # save images and

    if len(user_input) > 2:
        save_rooms_number(save_root,'save', rn)

        save_path = os.path.join(save_root, rn)
        url_room_num = 0

        while os.path.exists(save_path):
            save_path += '_' + str(url_room_num)
        os.makedirs(save_path, exist_ok=True)

        for i, num in enumerate(user_input):
            copy_path = os.path.join(imgs_path, imgs[num])
            save_image = os.path.join(save_path, f'{i}.jpg')
            shutil.copy(copy_path, save_image)
    else:
        save_rooms_number(save_root, 'del', rn)


# Main
if __name__ == '__main__':
    ## 경로 설정
    root = '/content/drive/MyDrive/AirbnbDataset/Images'   # 저장된 이미지 경로 (본인 환경에 맞게 수정 !!!)

    save_root = '/content/drive/MyDrive/AirbnbDataset/airbnb_images'   # 선택한 이미지 저장할 위치 (본인 환경에 맞게 수정 !!!)
    os.makedirs(save_root, exist_ok=True)

    pre_room_number = load_rooms_number(save_root)

    for it in sorted(os.listdir(root)):
        for rn in os.listdir(os.path.join(root, it)):
            if rn not in pre_room_number:
                for acc in os.listdir(os.path.join(root, it, rn)):
                    show_select_images(os.path.join(root, it, rn, acc), rn)