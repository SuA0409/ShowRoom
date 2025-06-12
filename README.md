## ShowRoom: Indoor Scene 3D Reconstruction System
Based on n user-selected images of a room, this system generates realistic 3D spaces by filling in missing 2D visual information (inpainting) and compensating for empty regions generated during the 3D reconstruction process.

## ShowRoom FlowChart
![image](https://github.com/user-attachments/assets/f706becb-5712-4f78-bd5b-499ded7b035c)


## Installation
    # clone project
    git clone https://github.com/SuA0409/ShowRoom
    cd ShowRoom

    # install requirements
    pip install -r requirements.txt

    # install ShowRoom as a package
    pip install -e .

## Usage
Demo dataset : demo/data
  
Loading Fast3R weight : 
  
※ The demo below only supports model execution and is not running on a server-based demo.
  ### Run to 3D Reconstruction (Fast3R-SPR-Viser)
  1.1. Fast3R
  
  1.2. SPR
  
  1.3. Viser

  ### Run to 2D Generation (Discriminator -> Generator)
  1.1. 2D discriminator
  
  1.2. 2D generator
  
## Project Structure
    ShowRoom/
    ├── model/               # 학습된 모델 및 구조
    ├── utils/               # 보조 함수들
    ├── inputs/              # 입력 이미지
    ├── outputs/             # 결과물 저장 위치
    └── README.md

## Fas3R KD 학습 방법과 결과
**1.1. Training data**
- Dataset

본 프로젝트는 ScanNet++ dataset을 활용하여 모델을 학습하였습니다.

**- Preprocessing Steps**

각 방(scene)에서 5장의 이미지를 선택합니다.

선택한 이미지들은 다음과 같은 전처리를 거칩니다:

    1. 크기 조절: 196 × 256

    2. 정규화: Min-Max Normalization

    3. Input 변환: Fast3R 모델에 맞는 입력 형태로 변환

**- 최종 Data Loader 설정**

각 샘플: 1개의 방(scene) = 5장의 이미지

    1. Batch size: 4

    2. 이미지 크기: [192 × 256 × 3]

    3. 전체 입력 Shape: [S, B, C, H, W]

        (S: 한 방에서의 이미지 수(5), B: 배치 내 방의 개수(4), C: 채널 수(3), H, W: 이미지 높이와 너비 (192, 256))
        
**1.2. Fast3R Input Format**

Fast3R은 각 view를 dictionary로 구성하고, 전체 scene은 dictionary들의 리스트로 표현합니다.

    [ {view_1}, {view_2}, ..., {view_S} ]  # List of S dictionaries
각 view_i는 다음과 같은 key-value 구조를 가집니다:

    image   : Tensor [B, 3, 192, 256]   # 정규화된 RGB 이미지
    true_shape : Tensor [B, 2]   # 원본 이미지 크기 정보      
    index   : list [B]   # 각 이미지의 인덱스       
    instance  : list [B]   # 방 인스턴스 ID        

(B는 배치(batch) 내의 방(scene)의 수)

(모든 view들은 같은 B를 공유하며, 총 S개의 view가 존재합니다 (예: 5장 이미지).)


**1.3. Knowledge Distillation of Fast3R**
**- Head 경량화:**

Student 모델에서는 Head의 hidden layer 수를 절반으로 줄여 파라미터 수를 줄이고 경량화를 달성했습니다.

**- Knowledge Distillation 손실 함수 구성:**

다음 세 가지 손실 함수를 사용하여 Teacher 모델로부터 효과적으로 지식을 전이하였습니다:

    1. RKD Distance Loss – 샘플 간 상대 거리 구조 유지

    2. RKD Angle Loss – 샘플 간 각도(구조적 관계) 보존

    3. Cosine Similarity Loss – Feature 간 방향 유사도 고려

세 손실 함수의 **단순 가중합(weighted sum)**을 최종 loss로 사용하였습니다.

**1.3. Train**
**- Scheduler 적용:**

학습 초반 안정화를 위해 초기 5 epoch 동안 learning rate를 1×10−4까지 warm-up한 뒤, 이후 학습이 진행되면서 점진적으로 1×10−6까지 감소시키는 스케줄러를 적용하였습니다.

**- Gradient Accumulation 적용:**

자원 제한으로 batch size를 4로 설정하였지만, gradient accumulation 기법을 활용하여 8회 반복 후 역전파를 수행, 결과적으로 batch size 32와 유사한 학습 효과를 얻도록 하였습니다.

**1.4. The Result of Train and Validation**

- Train: epoch에 따른 loss의 변화 (epoch 22까지의 진행상황)
![그림1](https://github.com/user-attachments/assets/dfc478fd-fef6-41ef-ad33-65d2e5f009e9)
- Validation:
  
  정량적 성능 평가
  ![image](https://github.com/user-attachments/assets/75ab280e-864f-42f9-909b-df12b4cb4b4d)

  정성적 성능 평가
  

## Citation
If you use this project or build upon it, please cite:
    
    @inproceedings{lin2023fast3r,
      title={Fast3R: Fast Room Reconstruction from a Single Panorama Using Transformers},
      author={Lin, Yen-Chen and Ranjan, Anurag and Liu, Lingjie and Tulsiani, Shubham and    Torralba, Antonio and Efros, Alexei A and Abbeel, Pieter and Freeman, William T and Zhang, Richard},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
    }
    
    @article{kazhdan2013screened,
      title={Screened Poisson Surface Reconstruction},
      author={Kazhdan, Michael and Hoppe, Hugues},
      journal={ACM Transactions on Graphics (TOG)},
      volume={32},
      number={3},
      pages={1--13},
      year={2013},
      publisher={ACM},
      doi={10.1145/2487228.2487237}
    }
    
    @article{gou2021knowledge,
      title={Knowledge Distillation: A Survey},
      author={Gou, Jianping and Yu, Baosheng and Maybank, Stephen John and Tao, Dacheng},
      journal={International Journal of Computer Vision},
      volume={129},
      number={6},
      pages={1789--1819},
      year={2021},
      publisher={Springer},
      doi={10.1007/s11263-021-01453-z}
    }
    
    @article{park2019relational,
      title={Relational Knowledge Distillation},
      author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
      journal={arXiv preprint arXiv:1904.05068},
      year={2019},
      doi={10.48550/arXiv.1904.05068}
    }


## License
This project is licensed under the [MIT License](./LICENSE)/
