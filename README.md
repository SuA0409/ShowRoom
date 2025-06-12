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
  ## Run to 2D Generation (Discriminator -> Generator)
  1.1. 2D discriminator
  
  1.2. 2D generator
  
## Project Structure
  ### vg
  Users can choose 3 ~5 images.
  ### gg
  The 3D space is reconstructed by sequentially applying Fast3R and SPR.
  ## gg

## Fas3R KD 학습 방법과 결과
    ### 1.1. train dataset 설명 및 전처리
    ### 1.2. Fast3R KD한 구조
    ### 1.3. Fast3R 학습방법
    ### 1.4. train/validation 결과

## Citation
    If you use this project or build upon it, please cite:
    
    @inproceedings{lin2023fast3r,
  title={Fast3R: Fast Room Reconstruction from a Single Panorama Using Transformers},
  author={Lin, Yen-Chen and Ranjan, Anurag and Liu, Lingjie and Tulsiani, Shubham and Torralba, Antonio and Efros, Alexei A and Abbeel, Pieter and Freeman, William T and Zhang, Richard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

## License
    MIT License

Copyright (c) 2025 SuA

Permission is hereby granted, free of charge, to any person obtaining a copy...

