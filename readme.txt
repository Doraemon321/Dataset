### Point cloud segmentation algorithm based on simulation assistance and neural network
Created by Jingxin Lin, Nianfeng Wang, and Xianmin Zhang, they are all from Guangdong Province Key Laboratory of Precision Equipment and Manufacturing Technology, South China University of Technology, Wushan Road, Tianhe District, Guangzhou, 510640, Guangdong, PR China.

### Introduction
This work is mainly used for point cloud segmentation of complex scenes, and the position and pose information of some objects in the scene are known. There are other objects in the scene whose information is unknown. If these two kinds of objects are close in position, the ordinary point cloud segmentation neural network may not be able to segment their point clouds well. However, if we model the object with known information and establish the simulated point cloud, through the comparison between the simulated point cloud and the actual point cloud, we can improve the segmentation performance and have better robustness and generalization.

It is common in industrial scenes that the position and pose information of some objects are known, such as the information of robots in the scene. So this method can be well applied in industrial safety detection or human-robot interaction. Of course, this method can not only be applied to industrial scenes. In our paper, we use a specific industrial scene to verify the method in the paper, and segment the points of the background objects, robots, people and other objects. The point cloud datasets used in the experiment are provided. The datasets include virtual dataset, real dataset and difficult dataset. A more detailed introduction will be in the "readme.txt" file in the "data" folder.

In order to vertify the effectiveness of the proposed method, we use some other state-of-art neural network point cloud segmentation methods for comparison. We also provide the code of these methods. Because the authors of these methods do not provide their source code, the code is reproduced according to their papers.

Under the "GADGCNN" folder is the test code of neural network GADGCNN (Cui, Y., Liu, X., Liu, H., Zhang, J., Zare, A., Fan, B.: Geometric attentional dynamic graph convolutional neural networks for point cloud analysis. Neurocomputing 432, 300¨C310 (2021)) in this experiment. Under the "point attension network" folder is the test code of point attension network (Feng, M., Zhang, L., Lin, X., Gilani, S.Z., Mian, A.: Point attention network for semantic segmentation of 3d point clouds. Pattern Recognition 107, 107446 (2020)) in this experiment. Under the "our network" folder is the test code of the proposed network in this experiment. 

The folder "fig" contains some related pictures.

### Installation
The code is tested under pytorch 1.7.1 and Python 3.8 (other version pytorch and Python may also work) on Windows 10. There are also some dependencies for a few Python libraries for data processing and visualizations like "numpy", "h5py" etc. You must have a GPU to run this code, and CUDA and CUDNN libraries are required.

Some CPP extension codes of pytorch need to be compiled. These codes are included under "torch_extension" folder. Run the following command in the "torch_extension" folder in the "cmd" environment to compile the codes:
```
	python setup.py install
```
Compiling this code requires some other C++ runtime libraries. If there are errors during compilation, installing VS2017 may solve the problem. 

### Train
You can train the corresponding network models by running the following command in different folders "GADGCNN", "point attension network" and "our network":
```
	python train.py
```
But you need to adjust the training parameters before training the model. There is a piece of code at the beginning of the "train" file to adjust the parameters, especially the three parameters of "BATCH_SIZE", "DATASET" and "model_name". The parameter "BATCH_SIZE" can be determined according to your GPU memory size. The parameter "DATASET" determines which dataset you will use to train the network. If the network is trained with real dataset or difficult dataset, it is recommended to fine-tune it on the basis of the network trained with virtual dataset, because the two datasets have a small number of point cloud scenes. The corresponding code has been added to the file. You don't need to do this manually. The parameter "model_name" is used to distinguish different trained models. For our network, because there are many different models, we need to modify the import statement at the top of the file to get different models.

You can also modify other parameters such as "NUM_POINT", "LEARNING_RATE" and so on. If you want to modify the network optimization method, modify it in the code in the middle of the "train" file.

After network training, the model with the highest segmentation accuracy in the training process will be saved in the "trained" folder, and the log in the training process will be saved in the "log" folder. At the same time, the loss, accuracy and mIOU of each epoch in the training process are saved in the "acc" folder. The file names of these files are determined by the parameter "model_name".

The network models that have appeared in the paper have a trained model in the "trained" folder, and the corresponding log files are also in the "log" folder and "acc" folder.

### Visualization of segmentation results
You can visualize the segmentation results of each data set by the corresponding network models by running the following command in different folders "GADGCNN", "point attension network" and "our network":
```
	python eval.py
```
But you need to adjust the parameters "MODEL_NAME" and "DATA_FILE" to select a network model to be tested and the point cloud dataset file you want to visualize before you run the code. These two parameters also appear at the beginning of the code. The code will create a new folder with the same name as "MODEL_NAME" under the "eval" folder, and the output visual point cloud data will be saved in this folder in ".txt" format. Each of these ".txt" files is the segmentation result of a scene point cloud, and different objects are represented in different colors. The point cloud data will be saved in the file in the form of N by 6 array, representing the three-dimensional coordinates and RGB color information of n points. Then you can use point cloud visualization software to read these point cloud files, such as CloudCompare.

### Evaluation
You can get the segmentation accuracy and mIOU of a trained model for a dataset by running the following command in different folders "GADGCNN", "point attension network" and "our network":
```
	python metrics.py
```
Similarly, you need to adjust the parameters "MODEL_NAME" and "DATA_FILE" to select a network and a dataset to be tested before you run the code. These two parameters also appear at the beginning of the code. In addition, since there are no AGVs or other flying obstacles in the difficult dataset, the union of these two types is 0 when calculating mIOU, which leads to calculation errors. Therefore, there is a "mask" parameter in the middle of the code to ignore the classes that do not need to be calculated. Similarly, since the percentage of background objects is too large and these points are easy to segment, you can also use this parameter to ignore the IOU of the background objects to avoid the high mIOU of all models, so as to show the difference of segmentation performance of each model in complex situations.

### Prepare your own data
If you want to apply our network structure to other specific scenes you need, you should first obtain the actual point cloud and simulated point cloud of the corresponding scene, and label the actual point cloud. Finally, these scene point clouds are combined into a ".h5" file, and you need to modify the corresponding parameters in the "get_data" file. If you want to use data files other than ".h5" format, please modify the "get_data" and "train" files to make the network read the dataset correctly.

### Acknowledgement
We achknowledeg that we borrow the code from [PointNet++](https://github.com/charlesq34/pointnet2) and [DGCNN](https://github.com/WangYueFt/dgcnn) heavily. Part of their code is mainly used in some files in the "torch_extension" folder. In addition, we also refer to the code from [pointSIFT](https://github.com/MVIG-SJTU/pointSIFT) when reproducing the point attension network.