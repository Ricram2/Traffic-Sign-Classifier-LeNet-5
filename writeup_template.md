
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/classdistribution.png "Class Distribution"
[image2]: ./examples/grayscale.png "Grayscaling"
[image4]: ./examples/test1.png "Traffic Sign 1"
[image5]: ./examples/test2.png "Traffic Sign 2"
[image6]: ./examples/test3.png "Traffic Sign 3"
[image7]: ./examples/test4.png "Traffic Sign 4"
[image8]: ./examples/test5.png "Traffic Sign 5"
[image9]: ./examples/Softmax1.png "Softmax1"
[image10]: ./examples/Softmax2.png "Softmax2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](hhttps://github.com/Ricram2/Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43

Also to render  signames.csv file for reference:
SignName
|Class ID|Description  |
|--|--|
|0|Speed limit (20km/h)|
|1|Speed limit (30km/h)|
|2|Speed limit (50km/h)|
|3|Speed limit (60km/h)|
|4|Speed limit (70km/h)|
|5|Speed limit (80km/h)|
|6|End of speed limit (80km/h)|
|7|Speed limit (100km/h)|
|8|Speed limit (120km/h)|
|9|No passing|
|10|No passing for vehicles over 3.5 metric tons|
|11|Right-of-way at the next intersection|
|12|Priority road|
|13|Yield|
|14|Stop|
|15|No vehicles|
|16|Vehicles over 3.5 metric tons prohibited|
|17|No entry|
|18|General caution|
|19| Dangerous curve to the left|
|20|Dangerous curve to the right|
|21|Double curve|
|22|Bumpy road|
|23|Slippery road|
|24|Road narrows on the right|
|25|Road work|
|26|Traffic signals|
|27|Pedestrians|
|28|Children crossing|
|29|Bicycles crossing|
|30|Beware of ice/snow|
|31|Wild animals crossing|
|32|End of all speed and passing limits|
|33|Turn right ahead|
|34|Turn left ahead|
|35|Ahead only|
|36|Go straight or right|
|37|Go straight or left|
|38|Keep right|
|39|Keep left|
|40|Roundabout mandatory|
|41|End of no passing|
|42|End of no passing by vehicles over 3.5 metric ... |

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed depending on each class. It is to be taken into consideration the uneven the data is, there are some classes with too little data. this may prevent the network to function more accurately.
![Number of images per Class][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to flatten the images to grayscale to make the dataset more manageable. colour images make it more processing consuming. **BIG DISCLAIMER:** Colour images processing could be rather a good choice for this model, because of the nature of traffic signs, colour can help describe better a class. 

Here is an example of a traffic sign image before and after grayscaling.

![Grayscaling][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							
| Grayscale 			| 32X32X1 Grayscale
| Layer 1 : Convolution 3x3 | 1x1 stride, same padding, outputs 32x32x64
| RELU					|												
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				
| Layer 2 : Convolution 3x3	| 1x1 srride, same padding, output 10x10x16
| RELU					|	
| Max pooling	      	| 2x2 stride,  outputs 5x5x16
| Flatten  				| output 400.        									
| Drop Out				| etc.        									
| Layer 3: Fully connected | Output = 120.        								
| RELU					|
| Layer 3: Fully connected | Output = 84. 
| RELU 					| 
| Layer 4: Fully connected | Output = 10
| Softmax				| 																				|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After adding 
To train the model, I used the following hyper parameters:

``Learning rate = 0.001
Epochs  = 10
Batch Size = 68``

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


The trick to make the accuracy at least a 0.93 was mostly relying on 2 things. 
1. Adding a Dropout to a LeNet architecture, and training with a  ``Keep probability = 0.8`` with this the network improved the Test Accuracy and prevented overfitting.
2. Making the Batch size higher also to improve the accuracy of the network. [idea backed by this paper](https://www.degruyter.com/downloadpdf/j/itms.2017.20.issue-1/itms-2017-0003/itms-2017-0003.pdf)
4. Setting a higher number of Epochs, after certain point it didn't show any improvement.

The final Hyperparameters were:

``Learning rate = 0.001
Epochs  = 100
Batch Size = 400
Keep probability = 0.8``
``

My final model results were:
`` Training: EPOCH 100 ...
Validation Accuracy = 0.944
Test Accuracy = 0.934``

Validation and test Accuracy were coherent and show no signs of overfitting or underfitting.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of the angle of the image
Te

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Main Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) 		| General caution 									| 
| Speed limit (60km/h)			| Speed limit (30km/h) 										|
| General caution					| General caution											|
| Keep right      		| Keep right					 				|
| Go straight or right		| Keep right    

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

| Image			        |      Second Best for missing ones        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) 		| No passing									| 
| Speed limit (60km/h)			| Speed limit (50km/h) 										|
| Go straight or right		| Go straight or right   							

If taken into coinsideration the second more probabble classes correctly mention the one more correct class


| Image			        |      Third Best for missing ones        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) 		| No passing for vehicles over 3.5 metric tons									| 
| Speed limit (60km/h)			| Speed limit (60km/h) 		

If taken into coinsideration the third more probabble classes correctly mention another correct class, leving the speed limit 30km/h misscalssified and not even in the next 5 probabble classes.
								
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

![alt text][image9] ![alt text][image10] 


