## One-line summary
Based on the n spatial images selected by the user, missing 2D visual information is complemented through inpainting, and empty regions generated during the 3D reconstruction process are refined to provide a realistic 3D space.

# ShowRoom FlowChart
![image](https://github.com/user-attachments/assets/f706becb-5712-4f78-bd5b-499ded7b035c)


## Installation
    # clone project
    git clone https://github.com/SuA0409/ShowRoom
    cd ShowRoom

    # install requirements
    pip install -r requirements.txt

    # install ShowRoom as a package
    pip install -e .

## The View of Developer
  Train dataset : ScnaNet++
  
  Loading Fast3R/Dust3R weight : 
  ### 1. Preprocess 2D images 
  1.1. Fixed-interval sampling : 5 images are sampled at regular intervals, with one image selected every (total/20) frames.
  
  1.2. Min-Max normalization : Each of the 5 images is resized to 196 × 256 and normalized to the range [-1, 1].
  
  1.3. Adjusting the input shape for model compatibility : The data is structured as a dictionary with the following keys: image, true_shape, index, and instance.
  ## 2. 3D Reconstruction
  We apply Knowledge Distillation(KD) of Fast3r. The teacher model is Fast3R (CVPR 2025), and the student model is ShowRoom.
  
  The type of KD used in the ShowRoom is relation-based knowledge and offline distillation.
  
  Change only the hidden layer number of head. (Loading encoder-decoder weight of original Fast3R's, and only train the weight of head)
  ## 3. Train
  [ 3.1. Dataset ]
  
  Approximately 500,000 samples from the ScanNet++ dataset are used with the following settings: 
  
  3.1.1. 5 sampled images per room
  
  3.1.2. a batch size of 4 rooms
  
  3.1.3. each image resized to 192 × 256 × 3
  
  The resulting data is reshaped to the format [S, B, C, H, W], where S is the number of images per room and B is the number of rooms in a batch.
  
  [ 3.2. Loss Function ]
  
  Calculate 'distance' and 'angle' between Fast3R(teacher) which is pseudo-GT(groud truth) and ShowRoom(student) output.
  
  Making loss more stable, using 'cosine loss'.
  
  (Applying normalization of each feature.)

  [ 3.3. Knowledge Distillation ]

  3.3.1. learning rate warm-up (schedular)

  3.3.2. Gradient accumulation 
  ## 4. Visualization
  [ SPR ]
  Used to complement the missing parts of the 3D reconstruction from Fast3R/ShowRoom (scene completion).
  
  As a result of comparing performance and execution time, SPR was executed twice with depth 9.
  
  [ Viser ]
  
  Using 'viser' which is the 3D plot tool made by Meta.
  
  Visulize 3D output in local using library.

  ## 5. Generate 2D images based on the input 2D images
  [ Discriminator ] by ST-RoomNet
  
  [ Generator ] by Stable-Diffusion-2-inpainting
  
## The View of Users
  ### 1. Choose 2D images
  Users can choose 3 ~5 images.
  ### 2. 3D Reconstruction
  The 3D space is reconstructed by sequentially applying Fast3R and SPR.
  ## 3. (Option) Choose the button '3D generate'
  
