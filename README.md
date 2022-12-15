# Multiple Object detection and classification for a Self Driving Car in an urban environment

- ## Project overview: 
The goal of this project is to create a convolutional neural network based model to detect and classify objects using data from the Waymo dataset.  By using a dataset of images of urban environments containing annotated cyclists, pedestrians and vehicles, the goal is to detect them in real-life test images and correctly classify them into one of the three categories.

- ## Set up: 
For the setup of the project, I used the project workspace provided by Udacity with the necessary data and libraries already available. 

1. The data to be used for the training was available in the project workspace but due to resource limitations of GitHub, I decided not to keep them in the current repository. Hence, an alternate method to create the necessary datasets is mentioned below -

The data for training, validation and testing can be organized as follow:

train: contain the train data
val: contain the val data
test - contains test files to test your model and create inference videos
The Waymo dataset downloaded from the internet needs to be split into train, val, and test sets by completing and executing the create_splits.py file (after modifying the function for implementing the split accordingly) or by manual splitting.


2. Experiments
The /home/workspace/experiments folder needs to be created and should be organized as follow:

pretrained_model
reference - reference training with the unchanged config file
exporter_main_v2.py - to create an inference model
model_main_tf2.py - to launch training
experiment0 - create a new folder for each experiment you run
experiment1 - create a new folder for each experiment you run
label_map.pbtxt

- ## Dataset:
After generating the bounding boxes around the objects of interest, some of the images from the ground truth data is displayed for visual inspection as follows -

![charts/display image 1.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%201.png)

![charts/display image 2.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%202.png)

![charts/display image 3.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%203.png)

![charts/display image 4.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%204.png)

- ### Dataset Analysis: 
The exploratory data analysis has been done on 7000 train as well as validation images to understand the distribution of objects belonging to different classes. The results obtained showed that there is a high imbalance among the classes of interest i.e. vehicle is the most overrepresented object class in the dataset whereas cyclists being the least represented. Pedestrians fall on a range between these two classes. The visualizated form of the results can be seen below - 

![class distribution in training image](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/training%20class%20distribution.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/validation%20class%20distribution.png)

- ## Training:
The config that we will use for this project is pipeline.config, which is the config for a SSD Resnet 50 640x640 model.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size.

A new config file called pipeline_new.config needs to be created in the /home/workspace/directory (can be modified as per convenience) by using the follwing command:

cd 'LOCATION WHERE THE edit_config.py IS LOCATED'

python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt

-Training Process:
The training process of the model can be run by - 

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

To monitor the training, a tensorboard instance can be launched by running - 

python -m tensorboard.main --logdir experiments/reference/ (location needs to be accordingly modified as per requirement)

-Evaluation Process:
Once the training is finished, launch the evaluation process.


python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/


(By default, it will run for one epoch.)


- ## Reference experiment: 
This section should detail the results of the reference experiment. It should include training metrics, Tensorboard charts, and a detailed explanation of the algorithm's performance.

- ## Improve on the reference: 
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

