## Graph Reasoned Multi-Scale Road Segmentation in Remote Sensing Imagery
(IGARSS 2023 - Awaiting Abstract Acceptance)

## Overview


## How to Run

### 1. Dataset Instructions
Download: <br>
[DeepGlobe](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) (Kaggle account required), <br> 
[MassachusettsRoads](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) (Kaggle account required), <br>
[Spacenet](https://spacenet.ai/spacenet-roads-dataset/) (AWS account required).

Once you download either the DeepGlobe or the Massachusetts Roads datasets, extract their contents into a "DeepGlobe" or "MassachusettsRoads" folder respectively in the Datasets folder.<br>

<details> 
  <summary>For Spacenet, the procedure is a bit more involved... </summary>
  
   <br> We need the images in 8-bit format.<br> After downloading AOIs 2-5 (Vegas, Paris, Shanghai, Khartoum), go to the [CRESI](https://github.com/avanetten/cresi) repository and select "SpaceNet 5 Baseline Part 1 - Data Prep".<br> Use [create_8bit_masks.py](https://github.com/avanetten/cresi/blob/main/cresi/data_prep/create_8bit_images.py) as described in the link. Then use [speed_masks.py](https://github.com/avanetten/cresi/blob/main/cresi/data_prep/speed_masks.py) to create continuous masks. Binarize these masks between [0,1] and place them in ```/Datasets/Spacenet/trainval_labels/train_masks/```

Next, locate the ```"PS-MS"``` folder in each corresponding ```AOI_#_<city>``` directory. <br>Move all image files in each of these "PS-MS" folders to ```/Datasets/Spacenet/trainval/```. <br>Like-wise, locate the ```"MUL-PanSharpen"``` folder in each corresponding ```AOI_#_<city>_Roads_Test_Public``` directory and move all of these image files to ```/Datasets/Spacenet/test/``` 
</details>

### 2. Setup

Create an environment with anaconda: ```conda create --name <your_env_name> python=3.9```<br>
Next, activate your environment: ```conda activate <your_env_name>```<br>
Install dependencies from pip: ```pip install -r requirements.txt```<br>
Install dependencies from conda: ```conda install pytorch=1.13.0 torchvision=0.14 pytorch-cuda=11.6 -c pytorch -c nvidia```

Now we will create our cropped images for each train/val/test part (where applicable) of a chosen Dataset.<br>
In the console enter: ```python setup.py -d Datasets -cs 512 -j <name of dataset>``` (-cs is the crop-size)<br>
The dataset name should be identical to the ones in the Dataset Instructions section. Wait approximately ~15 minutes.<br><br>
Cropped Image Disk Space:<br> DeepGlobe ~= 24.3GB<br> MassachusettsRoads ~= 9.71GB<br> Spacenet ~= 25GB<br>

### 3. Training

All training was performed on a single NVIDIA GeForce RTX 2080 Ti (11GB VRAM).<br>
See the ```cfg.json``` file to ensure that the training settings are appropriate for your rig. 
  
To train the model from scratch, run:<br>
```python train.py -m ConvNeXt_UPerNet_DGCN_MTL -d <dataset_name> -e <experiment_name>```<br>
<details>
<summary>Example</summary>
python train.py -m ConvNeXt_UPerNet_DGCN_MTL -d MassachusettsRoads -e MassachusettsRoads
</details>
  
To resume the training of a model:<br>
```python train.py -m ConvNeXt_UPerNet_DGCN_MTL -d <dataset_name> -e <experiment_name> -r ./Experiments/<experiment_name>/model_best.pth.tar```

To fine-tune a pre-trained model on a new dataset:<br> 
```python train.py -m ConvNeXt_UPerNet_DGCN_MTL -d <dataset_name> -e <experiment_name> -rd ./Experiments/<experiment_name>/model_best.pth.tar```<br>
For example, one can use pre-trained MassachusettsRoads model weights to start training for DeepGlobe or Spacenet to speed up convergence.

### 4. Evaluation
```Backup your log files (*.txt) in ./Experiments/<experiment_name>/```<br><br>
Once training ends (Default: 120 epochs), to evaluate Precision, Recall, F1, [IoU(relaxed)](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf) [IoU(accurate)](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf) metrics run:<br>
```python eval.py -m ConvNeXt_UPerNet_DGCN_MTL -d <dataset_name> -e <experiment_name> -r ./Experiments/<experiment_name>/model_best.pth.tar```

The evaluation script uses elements from the utils folder of [[3]](https://github.com/anilbatra2185/road_connectivity/tree/master/utils).
  
This will create a ```./Experiments/<experiment_name>/images_eval``` folder with each file showing (clock-wise) the original image, its label, a feature heat-map and the stitched prediction.
  
To evaluate the [APLS](https://github.com/avanetten/apls) metric refer to this [link](https://github.com/anilbatra2185/road_connectivity/issues/13).
  
### 5. Results
You may also refer to this [link](https://github.com/aavek/Satellite-Image-Road-Segmentation/blob/main/docs/IGARSS_Vekinis_2023_ea.pdf) for better viewing.
![results](https://user-images.githubusercontent.com/93454699/220936233-9be5869d-caf5-4723-af48-3a78bba6d91c.png)

<details> 
<summary>  6. REFERENCES </summary>
[1] N. Weir et al., “SpaceNet MVOI: A Multi-View Overhead Imagery Dataset”, 2019 IEEE/CVF International
Conference on Computer Vision (ICCV), 2019, pp. 992-1001, doi: 10.1109/ICCV.2019.00108.<br><br>
[2] I. Demir et al., “DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images”, 2018 IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2018, pp. 172-17209, doi:
10.1109/CVPRW.2018.00031.<br><br>
[3] A. Batra, S. Singh, G. Pang, S. Basu, C. V. Jawahar and M. Paluri, “Improved Road Connectivity by Joint Learning
of Orientation and Segmentation”, 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2019, pp. 10377-10385, doi: 10.1109/CVPR.2019.01063.<br><br>
[4] L. Zhang et al., “Dual Graph Convolutional Network for Semantic Segmentation”, 2019 British Machine Vision
Conference (BMVC), 2019, https://doi.org/10.48550/arXiv.1909.06121.<br><br>
[5] Z. Liu, H. Mao, C.Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, “A ConvNet for the 2020s”, 2022 Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11976-11986<br><br>
[6] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, J. Sun, “Unified perceptual parsing for scene understanding”. In: Ferrari, V.,
Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) ECCV 2018. LNCS, vol. 11209, pp. 432–448. Springer, Cham
(2018). https://doi.org/10.1007/978-3-030-01228-1_26<br><br>
[7] A. Etten, D. Lindenbaum, T. Bacastow, “SpaceNet: A Remote Sensing Dataset and Challenge Series”, 2018,
https://doi.org/10.48550/arXiv.1807.01232<br><br>
[8] V. Mnih, “Machine Learning for Aerial Image Labeling”, PhD Dissertation, University of Toronto, 2013.<br><br>
[9] W.G.C. Bandara, J.M.J. Valanarasu, V.M .Patel, “Spin road mapper: extracting roads from aerial images via spatial
and interaction space graph reasoning for autonomous driving”. arXiv preprint arXiv:2109.07701 (2021)
</details>
  
