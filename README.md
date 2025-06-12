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
### 1.1. train dataset 설명 및 전처리
### 1.2. Fast3R KD한 구조
### 1.3. Fast3R 학습방법
### 1.4. train/validation 결과

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
