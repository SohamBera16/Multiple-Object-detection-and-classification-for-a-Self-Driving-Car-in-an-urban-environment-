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

- Training Process:
The training process of the model can be run by - 

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

To monitor the training, a tensorboard instance can be launched by running - 

python -m tensorboard.main --logdir experiments/reference/ (location needs to be accordingly modified as per requirement)

- Evaluation Process:
Once the training is finished, launch the evaluation process.


python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/


(By default, it will run for one epoch.)


- ## Reference experiment: 
For the reference experiment using the baseline model, the default configurations according to the file pipeline.config has been implemented. The baseline ssd resnet50 model's detection capacity on the validation dataset was poor with classification loss after 2.5k training steps being approximately 0.6. Some of the other metrics and corresponding charts have been shown below - 

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20baseline%201.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20baseline%202.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20baseline%203.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20baseline%204.png)

- ## Improve on the reference: 
Different strategies that have been adopted to improve your model performance include several data augmentation techniques and changes in the learning rate, namely, 

1. random_rgb_to_grayscale conversion with probability 0.2
2. random distort color 
3. random adjust brightness
4. random adjust contrast
5. random adjust saturation
6. random adjust hue 
7. changing the base and warmup learning rates to 0.0005

Some of the examples of augmentation in the training images looks as follows - 

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/augmented%20image%201.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/augmented%20image%202.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/augmented%20image%203.png)


As a result of the changes, the classification loss metric score went down around 0.18 from 0.6 (almost 40% drop in loss). Also, the detections on the test dataset was much more plausible whereas the baseline model was not able to detect most of the objects. Some of the charts as depicted here - 

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20improved%201.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20improved%202.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20improved%203.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/train%20improved%204.png)

- ## Final Results:
Some of the snippets from the animations generated by running inference using the improved model on test dataset looks as follows -

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/test%20result%201.png)

![class distribution in val images](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/test%20result%202.png)

