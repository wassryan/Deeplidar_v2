# DeepLiDAR implementation_v2
This repository contains the code (in PyTorch) for "[DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qiu_DeepLiDAR_Deep_Surface_Normal_Guided_Depth_Prediction_for_Outdoor_Scene_CVPR_2019_paper.pdf)" paper (CVPR 2019).

**Note**: 
- I modify the dataloader using hdf5 file to make data reading faster. 
- I list the whole process much clearly from `generating surface normal` to `train` by hdf5 file
- I modify some details to make it run successfully, and I will show my result below.

Here is the [Original Code](https://github.com/JiaxiongQ/DeepLiDAR) provided by author.

## Introduction
In this work, we propose an end-to-end deep learning system to produce dense depth from sparse LiDAR data and a color image taken from outdoor on-road scenes leveraging surface normal as the intermediate representation.
![image](https://github.com/JiaxiongQ/Need2Adjust/blob/master/pipline.PNG)
## Requirements
- [Python2.7](https://www.python.org/downloads/)
- [PyTorch(0.4.0+)](http://pytorch.org)
- torchvision 0.2.0 (higher version may cause issues)
- [KITTI Depth Completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
### Pretrained Model
※NOTE: The pretrained model were saved in '.tar'; however, you don't need to untar it. Use torch.load() to load it.

[Download Link](https://drive.google.com/file/d/1eaOCtl_CGzqqqJDbVawsdniND255ZaP8/view?usp=sharing)

## Generate Surface Normal
※NOTE: (**Requirements**) g++ 4.1/ ubuntu16.04/ opencv2.4.9(I prefer using docker to complete this task)
```
cd surface normal
make
./count_main.app
```
The surface normal generated from `gt depth` is under the `./normal_data`

## Train
Use the training strategy in the folder named 'trainings'.
1. Generate .h5 file
`python ./dataloader/data2h5.py`，
2. Start to train using the modified code for .h5 file
```
python h5trainN.py --batch_size 2 --gpu_nums 2 --epochs 20
python h5trainD.py --loadmodel N_model_20.tar --batch_size 2 --gpu_nums 2
python h5train.py --loadmodel D_model_20.tar --batch_size 2 --gpu_nums 2
```
It cost a lot of time to train, and the final model will be generated as `Final_model_20.tar`

## Evaluation
1. Change the names of the folders in 'test.py' by yourself:
```
'gt_fold': the location of your groundtruth folder;
'left_fold': the location of your RGB image folder;
'lidar2_raw': the location of your Sparse(LiDAR) depth folder.
```
 
2. Use the following command to evaluate the trained on the test data.
```
python test.py --loadmodel Final_model_20.tar
```
3. I also provide script to show the result in color and you can also choose to save the result
```
python test_show_result.py --loadmodel Final_model_20.tar
```
## My result
The whole process of training: h5trainN.py(6 epoches)+h5trainD.py(12 epoches)+h5train.py(12 epoches), and the result wasn't good(maybe due to the fewer epoches)

![result1](https://github.com/k-miracle/Deeplidar_v2/blob/master/result1.png)
![result2](https://github.com/k-miracle/Deeplidar_v2/blob/master/result2.png)
