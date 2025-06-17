## ShowRoom: Indoor Scene 3D Reconstruction System
Based on n user-selected images of a room, this system generates realistic 3D spaces by filling in missing 2D visual information (inpainting) and compensating for empty regions generated during the 3D reconstruction process.

## ShowRoom FlowChart
![image](https://github.com/user-attachments/assets/81453487-72b1-4635-b23e-1e375079e726)



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

※ The demo below only supports model execution and is not running on a server-based demo.
  ### Run to 3D Reconstruction (Fast3R → SPR → viser)
      python demo/show_room_demo.py

  ### Run to 2D Generation (ST-RoomNet (discriminator) → Stable-Diffusion (generator))
  Download the weight file from the following link: https://drive.google.com/file/d/1j2eQdEMWsHPpULlGBkZxVO6QFeOM0E1E/view?usp=sharing
  
  Put the weight file into the weight folder inside the generator_2d folder.
  
      python demo/generator_2d_demo.py

If you want to run the demo with your own image, put your image and pose in the demo/data directory.

The generated images are saved in the output folder.

  ## Run to Review
      python demo/review_demo.py —url [Airbnb URL]   # Enter the URL of the Airbnb listing as a string to extract the review topics.
      python demo/review_demo.py


      
      
## Project Structure
    ShowRoom/
    ├── .idea/                         # PyCharm project settings
    ├── EDA/                           # Exploratory Data Analysis scripts
    ├── chrome_extension/              # Chrome extension implementation code
    ├── configs/                       # KD model training and environment configuration files
    ├── data/                          # Preprocessing for training data
    │   └── scannet/
    ├── demo/                          # Demo code and sample data for ShowRoom
    │   ├── data/
    ├── fast3r/                        # Fast3R (CVPR 2025); teacher model of KD
    ├── generate2d/                   
    │   ├── discriminator/             # Discriminator of 2D image
    │   └── generator/                 # Generator of 2D image
    ├── kd_fast3r/                     # Knowledge Distillation training modules for Fast3R
    │   └── utils/
    ├── review/                        # Visualization and review of model evaluation results
    ├── server/                        # Flask-based backend server
    │   ├── templates/
    │   ├── results/
    │   └── static/
    ├── viz/                           # Visualization modules
    ├── LICENSE                        # License information
    ├── README.md                      # Project overview and documentation
    ├── requirements.txt               # Package dependency list
    └── setup.py                       # Installation script


## Knowledge Distillation of Fast3R
**1.1. Training Dataset**

This project was trained using the **ScanNet++ dataset**.

**1.2. Preprocessing Steps**

For each scene, **5 images** were selected and the following preprocessing steps were applied:

1. Resizing: 196 × 256

2. Normalization: Min-Max Normalization

3. Input Conversion : Transform to match the input format of the Fast3R model

**Fast3R Input Format**

Each scene was represented as a list of dictionaries, one per view:

    [ {view_1}, {view_2}, ..., {view_S} ]  # List of S dictionaries

Each view_i contained the folling keys:

    image   : Tensor [B, 3, 192, 256]   # noramlized RGB image 
    true_shape : Tensor [B, 2]   # original image dimensions
    index   : list [B]   # image indices 
    instance  : list [B]   # scene instance IDs 

B refers to the number of scenes in a batch. All views shared the same batch size B, and there were a total of S views per scene (e.g. 5 images)

**- Final DataLoader Settings**

1. Batch size: 4

2. Image size: [192 × 256 × 3]

3. Final input shape : [S, B, C, H, W]
    
   S: Number of views per scene (= 5),
   
   B: Number of scenes per batch (= 4),
   
   C: Number of channels (= 3),
   
   H, W: Height and width (= 192, 256)

**1.3. Knowledge Distillation of Fast3R**

**- Lightweight of Head:**

In the student model, the number of hidden layers in the head was reduced by 50% to decrease parameter size and achieve model compression.

**- Distillation Loss:**

To effectively transfer knowledge from the teacher model, we used a weighted sum of the following three loss functions:

1. RKD Distance Loss : Preserving relatvie distance structure between samples

2. RKD Angle Loss – Maintains angular relationships betweeen samples

3. Cosine Similarity Loss – Encourages directional similarity between feature vectors

The final loss was computed as a simple weighted sum of the three. 

**1.4. Train**

**- Learning Rate Scheduler:**

A warm-up scheduler was applied to stabilize early training.

- Learning rate increased to 1×10−4 during the first 5 epochs
- Gradually decayed to 1×10−6 over subsequent epochs

**- Gradient Accumulation:**

Due to memory constraints, we used a batch size of **4**. To simulate a larger effective batch size of **32**, we applied **gradient accumulation**, updating the model parameters every **8 steps**.

**1.5. The Result of Train and Validation**

- Train: Loss trends across epochs (up to 22 epoch)
  ![image](https://github.com/user-attachments/assets/005acab3-5db1-42f2-b4ea-a34d5131ee01)



- Validation:
  
  - Qualitative performance evaluation
  ![image](https://github.com/user-attachments/assets/a95896f1-0d89-437b-8d1d-415244395c11)

  - Quantitative performance evaluation
  ![image](https://github.com/user-attachments/assets/3e51b04d-be5b-4ed3-bb75-12a5321e035b)


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

    @misc{rombach2021ldm,
      title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
      author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
      year      = {2021},
      month     = {December},
      eprint    = {2112.10752},
      archivePrefix = {arXiv},
      url       = {https://arxiv.org/abs/2112.10752}
    }

    @inproceedings{ibrahem2023stroomnet,
      title   = {ST-RoomNet: Learning Room Layout Estimation From Single Image Through Unsupervised Domain Adaptation},
      author  = {Ibrahem, Mohammed and Yuan, Ye and Shen, Yilun and Hoai, Minh},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
      year    = {2023},
      url     = {https://openaccess.thecvf.com/content/CVPR2023W/VOCVALC/html/Ibrahem_ST-RoomNet_Learning_Room_Layout_Estimation_From_Single_Image_Through_Unsupervised_CVPRW_2023_paper.html}
    }

    @misc{park2019relational,
      title   = {Relational Knowledge Distillation},
      author  = {Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
      year    = {2019},
      archivePrefix = {arXiv},
      eprint  = {1904.05068},
      url     = {https://arxiv.org/abs/1904.05068}
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

## License
This project is licensed under the [MIT License](./LICENSE)/
