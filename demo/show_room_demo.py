from kd_fast3r.kd_fast3r import ShowRoom
from viz.viz import ViserMaker
from kd_fast3r.utils.data_preprocess import batch_images_load

# --1. 모델 선언
 # model과 viser 선언
show_viz = ViserMaker()
model = ShowRoom(viz=show_viz)

# --2. 데이터 로드
 # Input 이미지 폴더의 주소
folder_path = "demo/data"  # Input Your Name of Image Folder
 # Input 이미지의 개수
number_of_image = 3
 # 이미지 전처리
model.room = batch_images_load(rooms_path=folder_path, batch_size=1, size=512, sample=number_of_image)

# --3. 모델 추론
 # fast3r, spr 그리고 viser의 결합
model.reconstruction()
model.building_spr()
