#**Behavioral Cloning**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[architecture]: ./figs/cnn-architecture.png "CNN Architecture"
[center_example]: ./figs/center_example.jpg "Center example"
[left_example]: ./figs/left_example.jpg "Left example"
[right_example]: ./figs/right_example.jpg "Right Image"
[cropped]: ./figs/cropped.png "Cropped Image"
[brightness]: ./figs/brightness.png "Changing brightness"
[flipped]: ./figs/flipped.png "Flipped"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* *train_model.ipynb* containing the script to create and train the model.
* *drive.py* for driving the car in autonomous mode.
* *model.h5* containing a trained convolution neural network.
* *README.md* summarizing the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The *train_model.ipynb* file contains the code I used to train a convolutional
neural network used for autonomous driving. The sections in the notebook are:
- Utility functions (used to read in images from the training dataset and images
augmentation.)
- Validation set preparation.
- Training set preparation.
- Model training.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

![alt text][architecture]

I use the architecture designed by NVIDIA (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
The model consists of one normalization layer, four convolutional layers, and
four fully-connected layers.

Each convolutional and fully-connected layer is followed by an ELU activation layer
(to add nonlinearity to the model).


#### 2. Attempts to reduce overfitting in the model

After each ELU layer I add a dropout layer. The drop rate is set at 0.2.

The validation set is prepared by getting images from driving two laps in each track in the simulator.
The same validation set is used while trying different architectures and hyperparameters.
Therefore, I could easily compare the performance between different models.

To test a trained model, I use that model to drive a car in the simulator (both tracks).


#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually.
I did not try different values for dropout rate.

#### 4. Appropriate training data

Training data consists of:
- Center lane driving in both tracks and in both directions (clockwise and counter-clockwise), recorded by the center camera.
- Recovering from the left and right sides of the road on both tracks.
- Augmenting data by randomly changing brightness in the images, and flipping the images.
- Images recorded by left and right cameras.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The NVIDIA neural network architecture is introduced during lecture, and it is proven to work well, so I decide to use it as the architecture for my model.

To obtain training data, I drive one lap in each direction for both tracks provided in the simulator. I also do some recovery driving by taking the car to the left or right side then steer it back to the center of the lane.
It was not very easy driving with the simulator because the driving feeling is totally unrealistic. I think the data get better when I turn smoothly on corners, i.e. try to keep the steering angle consistent during a turn.

I loaded the training data with images of left, center, and right cameras using *matplotlib*. It is important to note that in the example provided during lecture, images are imported using *OpenCV*. Since OpenCV uses BGR colormap as default, it affects the accuracy of the model because the model uses RGB colormap during testing.

The validation set is obtained separately from the training set because I want the validation set to be consistent so I can easily compare different networks. The validation set consists of two laps of driving in each track.

With the training and validation sets available, I use Keras to train a convolutional neural network with the same hyperparameters used by NVIDIA. After raining is finished, the model can be used to drive in the simulator by using the script *drive.py*.


#### 2. Final Model Architecture

added later using a table

#### 3. Creation of the Training Set & Training Process

In this section I will show examples of the training data and some image augmentation.

Here is an example of center-lane driving:
![alt text][center_example]

The car is also equipped with left and right cameras, and the images recored from these cameras look like (from the same position above):

![alt text][left_example]

![alt text][right_example]

Because we assume all images come from the center camera, when use the images from the left and right cameras, I need to add some steering angle correction to the recorded steering angle. In other words, images from left and right cameras can simulate the situation when the car is on either side, and we need to steer it back to the center.

I also do some recovery driving, i.e. I take the car to the side then drive it back to the center. It was a great way to add more meaningful data to the training set.

I repeat this process to obtain the validation set. The important difference is only images from the center camera are used, and no recovery driving is recorded.

One important preprocessing step is to crop the image so that the image only contains the road instead of other artifacts in the scene (like trees, water, etc.). This cropping step is done via a layer in the Keras model. Here is an example of a cropped image.

![alt text][cropped]

Next, I try to augment the training set so the model can generalize better (reduce overfitting). I use two methods of augmentation: changing brightness and flipping the images.

The brightness is changed by converting a RGB-image to HSV colormap, changing the value of the V-channel, and converting it back to RGB colormap. Here is example of decreasing the brightness of an image.

![alt text][brightness]

Flipping an image is done via the *fliplr* function of numpy. When we flip the image we also multiply the steering angle by -1 (flipping the sign).

![alt text][flipped]

In summary, the training set has about 60000 images while the validation set has about 7000 images. The model is trained using Keras and saved as the file *model.h5* in the repo.

#### 4. Misc. Notes

In the drive.py file, the steering angle is predicted by the model, then I multiply it by 1.5. Otherwise, the car does not steer hard enough during sharp turn. 
