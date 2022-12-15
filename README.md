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
![charts/display image 1.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%201.png)

![charts/display image 2.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%202.png)

![charts/display image 3.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%203.png)

![charts/display image 4.png](https://github.com/SohamBera16/Multiple-Object-detection-and-classification-for-a-Self-Driving-Car-in-an-urban-environment-/blob/main/charts/display%20image%204.png)

- ### Dataset Analysis: 
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts, and other visualizations.

- ## Training:

- ## Reference experiment: 
This section should detail the results of the reference experiment. It should include training metrics, Tensorboard charts, and a detailed explanation of the algorithm's performance.

- ## Improve on the reference: 
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

