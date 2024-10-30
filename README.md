# Energy-based Domain Adaptation without Intermediate Dataset for Foggy Scene Segmentation
This work was accepted at IEEE Transactions on Image Processing (TIP). [Paper](https://ieeexplore.ieee.org/abstract/document/10735117)
## Requirements
This repository is implemented on
+ **Ubuntu 16.04**
+ **Conda 4.12.0**
+ **CUDA 11.6**
+ **Python 3.7.15**
+ **Pytorch 1.12.0**



To install required environment for the training:
```
conda env create -n [your env name] -f environment.yaml
conda activate [your env name]
```

You need to install [densetorch](https://github.com/drsleep/densetorch) package. 
To install densetorch:
```
git clone https://github.com/drsleep/densetorch
cd densetorch
pip install -e .
```


## Dataset
You can download the dataset here:

+ **Cityscapes**: Download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from the [Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put it in the './data/Cityscapes' directory.

+ **Foggy Zurich**: Download "Foggy Zurich.zip" from the [Foggy Zurich Dataset](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/), and put it in the './data/Foggy_Zurich' directory.

+ **Foggy Driving**: Downlaod "Foggy Driving.zip" from the [Foggy Driving Dataset](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/), and put it in the './data/Foggy_Driving' directory.

+ **ACDC**: Download "rgb_anon_trainvaltest.zip" and "gt_trainval.zip" from the [ACDC Dataset](https://acdc.vision.ee.ethz.ch/), and put it in the './data/ACDC' directory.


## Pre-trained Models

You can download pretrained model here:

+ **Cityscapes pre-trained model**: [./Cityscapes_pretrained_model.pth](https://drive.google.com/file/d/1SHUwqKAqcez6HFb93f-VSBy1R4b6keqb/view?usp=drive_link)

+ **Pre-trained DAEN**: [./DAEN_1.pth](https://drive.google.com/file/d/1pE2010C-chzOVcFgRbpLK_AZ-3QWQv_W/view?usp=sharing)


## Evaluation
To evaluate DAEN on real foggy datasets:

```
CUDA_VISIBLE_DEVICES=[gpu_id] python evaluate.py --file-name 'DAEN_result' --restore-from './DAEN_1.pth'
```

## Training 
To train DAEN:
```
CUDA_VISIBLE_DEVICES=[gpu_id] python main.py --file-name 'DAEN' --restore-from './Cityscapes_pretrained_model.pth'
```


## Acknowledgements
This code is built on [FIFO](https://github.com/sohyun-l/fifo). We thank the authors for sharing their codes.

