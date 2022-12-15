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

- ## Dataset:

- ### Dataset Analysis: 
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts, and other visualizations.

- ## Training:

- ## Reference experiment: 
This section should detail the results of the reference experiment. It should include training metrics, Tensorboard charts, and a detailed explanation of the algorithm's performance.

- ## Improve on the reference: 
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

