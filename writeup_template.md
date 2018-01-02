#**Traffic Sign Recognition** 

##Nate Yoder Writeup

---

**Traffic Sign Recognition Project**

As part of this project I focused on the following areas:
* Exploring, summarizing and visualizing the data set
* Data set augmentation
* Designing, training, refining and testing a model architecture
* Use the final model to make predictions on new images
* Analyzing the softmax probabilities of the new images


[//]: # (Image References)

[signs]: ./output/AllSignsExamples.png
[class_count]: ./output/ClassCountByStage.png
[augmented]: ./output/AugmentedImages.png
[sign3]: ./examples/30kph_sign.jpg "Traffic Sign 3"
[sign5]: ./examples/construction_sign.jpg "Traffic Sign 5"
[sign4]: ./examples/straight_right_sign.png "Traffic Sign 4"
[sign1]: ./examples/stop_sign.jpg "Traffic Sign 1"
[sign2]: ./examples/100kph_sign.jpg "Traffic Sign 2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary

 Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate some basic summary statistics of the traffic
signs data set:

* The number of unique classes/labels in the data set is 43
* The size of training set is 34,799
  * The least common classes (class 0, 37, and 19) had 180 examples, 
  25% of the labels had less than 285 examples, 
  50% of the labels had 540 examples or less, 
  and the most common class (class 2) had 2010 examples.
* The size of the validation set is 4,410 images
  * The least common classes (class 0, 39, 37, 32, 29, 27, 24, 41, 19, and 42) had 30 examples,  
  50% of the labels had 60 examples or less, 
  and the most common classes (class 1 and 2) had 240 examples.
* The size of test set is 12,630 images
  * The least common classes (class 0, 37, 32, 27, 41, and 19) had 60 examples,  
  50% of the labels had 180 examples or less, 
  and the most common classes (class 2) had 750 examples.
* The shape of a traffic sign image is 32 pixels square with 3 "channels" for rgb colors


To get a fuller picture of the distribution of each of the classes see the barchart below.

![alt text][class_count]

### Exploratory visualization of the dataset.

As an initial step in exploring the signs to be classified I visualized an example sign from each of the classes. 
One aspect of the data that became clear during this process is that the brightness of the images seems to vary 
significantly in the training data. I also noted that several of the signs seemed to be slightly rotated. 

![alt text][signs]

###Design and Test a Model Architecture

####Image augmentation

Because of the relatively limited amount of data in some classes and some of the image variation seen in the data 
set I explored image augmentation strategies.  Therefore, rather than normalizing the images up front I left the 
images as being colored and modified them using the [imgaug](https://github.com/aleju/imgaug) python package.

In order to augment the images I performed an affine transformation and then added gaussian noise to the image. The 
affine transform included:
* Randomly scaling the image between 85-120% in each direction
* Randomly translating the images between -15 and 15% in each direction
* Randomly rotating the image between -12 and 12 degrees
* Randomly shearing the image between -4 and 4 degrees
* Performing interpolation using bilinear interpolation
* Adding a small amount standard deviation of 3% of random gaussian noise
* Randomly scaling the brightness of the image by a factor between 1/2.0 and 1.5
* Randomly perturbing the colors by scaling each channel by a factor between 1/1.2 and 1.2

Here is an example of an original image and seven randomly augmented versions of that image:

![alt text][augmented] 

This augmentation process was performed while the model was being trained so that each epoch would include a 
slightly different version of the training images. This likely slowed the training process but should have helped 
reduce potential overfitting. During the first epoch no augmentation was performed so that at least one unaugmented 
version of the images would be seen during the training process. After the image augmentation was performed the images were then normalized by simply diving the RGB values by 255 to
 normalize them between 0 and 1.

####2.Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, kernel size 5, 14 filters, outputs (28x28x14)
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x14 	    			|
| Convolution 5x5	    | 1x1 stride, valid padding, kernel size 5, 30 filters, outputs (10x10x30)      			
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Fully connected		| 300 outputs         							|
| RELU					|												|
| Dropout				| Keep probability = 0.6	                    |
| Fully connected		| 150 outputs         							|
| RELU					|												|
| Dropout				| Keep probability = 0.6						|
| Softmax				| 43 outputs        							|
 


####3. Model training

The first step in training the model was to determine how to initialize the weights and biases. 
I initialized the weights using a truncated normal distribution with a mean of zero and a standard deviation of 0.1.
I initialized the biases of layers before the the RELU activation functions to be slightly positive (1E-3) in order to 
bias them towards activating the neuron. I set the biases of the output layer to be equal to the class probabilities .
I used an ADAM optimizer with an epsilon of 1E-6.
I tried several different batch sizes and looked at the amount of time per epoch. 
I settled on a batch size of 256 as that seemed to give reasonably fast convergence.
I tried several different learning rates in an ad hoc fashion but eventually settled on a learning rate of 0.001. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 97.3%
* test set accuracy of 96.3%

I started from the LeNet classifier provided in class. 
However, because I thought colors might be important I decided to maintain the RGB color spectrum.
Because of this increase in the dimensionality of the input and the much larger number of potential output classes I
 increased the sizes of each of the layers pretty substantially to ensure that I was able to obtain very high 
 accuracies on the testing dataset. 
I  then relied on the use of Dropout and image augmentation to help me ensure that I still obtained acceptable 
 validation error and did not overfit the data too much. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3] 
![alt text][sign4] ![alt text][sign5]

The first step after loading these images was to scale all of these so they 
were 32 by 32.

The first sign is a stop sign and might be difficult due to the other sign below it and the trees in the background.

The second sign is a 100 kph sign but the image is taken from below the sign so that might be a little difficult.

The third sign is a 30 kph sign and is a bit off center and has a more complicated background than most of the others.

The fourth sign is a arrow sign (straight & right) and might prove more difficult because it had a blank 
background.

The last sign is a contruction sign but it has small attachments o the top and bottom along with a slightly upwards 
angle which might make it a bit more challanging.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 100 km/h   			| 100 km/h 										|
| 30 km/h				| 30 km/h										|
| Straight & Right	    | Straight & Right				 				|
| Construction			| Construction      							|


The model was able to correctly guess all 5 of traffic signs, which gives an accuracy of 100%. While these results 
are promising this is a very small test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making looking at the softmax probabilities using my final model is located in the 70th cell of the 
Ipython notebook.

For all of these examples the model was almost too sure about it's prediction.  
In each case the probability it assigned to the top prediction was over 99.9%.

For the first image, the model is confident that 
it is a stop sign (probability of >99.9), and the image does contain a stop sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.99       			| Stop sign   									| 
| 0.01     				| Do not enter 									|
| <0.01					| Yellow diamond								|
| <0.01	      			| Bump					 		        		|
| <0.01				    | Yield               							|


For the second image, the model is confident that 
it is a 100 km/h (probability of >99.99) sign and it is correct. 
Other speed limit signs are also in the top 5 but have low probabilities.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >99.99       			| 100 km/h    									| 
| <0.01    				| 80 km/h|
| <0.01					| 120 km/h|
| <0.01	      			| Truck|
| <0.01				    | Class 30               							| 


For the third image, the model is confident that 
it is a 30 km/h (probability of >99.99) sign and it is correct. 
Other speed limit signs are also in the top 5 but have low probabilities.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >99.99       			| 30 km/h    									| 
| <0.01    				| 20 km/h|
| <0.01					| 50 km/h|
| <0.01	      			| 70 km/h|
| <0.01				    | 100 km/h|

For the fourth image, the model is confident that 
it is a straight & right sign (probability of >99.99) sign and it is correct.   

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >99.99       			| straight & right   									| 
| <0.01    				| Straight|
| <0.01					| Lower right|
| <0.01	      			| Left turn|
| <0.01				    | Class 11|

For the fifth image, the model is confident that 
it is a construction sign (probability of >99.99) sign and it is correct.
The last two values are zero and so are simply the lowest two classes.   

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >99.99       			| Construction   									| 
| <0.01    				| Class 30|
| <0.01					| Class 29|
| <0.01	      			| 20 km/h|
| <0.01				    | 30 km/h|

