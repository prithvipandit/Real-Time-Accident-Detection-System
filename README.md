# Real-Time-Accident-Detection-System

### An Accident Detector on IOT Devices

#### Table of Content
[Summary](#summary)</br>
[Dataset Generation](#datasetGeneration)</br>
[Data Processing](#dataProcessing)</br>
[Model Training](#modelTraining)</br>
[Pre-Trained Mode](#preTrainedModel)</br>
[References](#ref)</br>


### Summary<a name="summary"></a>
The model implements CNN(Convolutional Neural Network)-based learning on accident footages captured through traffic cameras. The model aims to detect Real-time Accident via an IoT-based traffic camera. 
<br/>         


### Data Generation<a name="datasetGeneration"></a>
For the dataset generation, We use the CADP Dataset which comprises a collection of accident videos and DETRAC dataset containing footages of normal traffic flow. Initially, we supposedly marked the 'CADP' as an Accident Dataset and 'DETRAC' as a non-accident set. Since the CNN model ought be applied on images, rather than videos, we converted the video footages into images(using frame_cutter.py). Training on this 'Crude Dataset', the initial training and validation error was significant, partly due to the large number of 'False Negative' in the 'Accident Dataset'. In order to overcome this, we manually removed the false negatives from 'Accident Dataset'. Finally, we obtained the final 'Compact Dataset' with 4000 accident images from 'Accident Dataset' and 8000 images from 'Non-accident Dataset' (12000 images overall). The 'Compact Dataset' can be found here Compact Dataset. In the next section, we describe the Data Preprocessing employed before training the model. 

<br/> 

### Data Processing<a name="dataProcessing"></a> 
We converted each video frame into an image. Each of these images is a two-dimensional array of pixels, where each pixel has information about the red, green, and blue (RGB) color levels. To reduce the dimensionality at the individual image level, we convert the 3-D RGB color arrays to grayscale. Additionally, to make the computations more tractable on a CPU, we resize each image to (128, 128) - in effect reducing the size of each image to a 2-D array of 122 X 128. 

<br/>

### Model Training<a name="modelTraining"></a> 
We built a convolutional neural network for image classification with keras.

We created a sequential model which linearly stacks all the the layers, using keras.models. We implemented different keras layers like Conv2D- convolutional layer, MaxPooling2D- max pooling layer, Dropout, Dense- add neurons, Flatten- convert the output to 1D vector, using keras.layers along with 'ReLU' and 'softmax' as activation functions.

While compiling the model we used "sparse_categorical_crossentropy" as our loss function, "adam" as our optimizer and "accuracy metrics" as metrics. Then in model fit we ran the model for "10" epochs with "0.2" validation_split.

Using this model we are able to predict whether given video contains accident or not.


### Code Requirements
You can install [PyChram](https://www.jetbrains.com/pycharm/) for python which resolves all dependencies needed to run this Project


#### Procedure
1. Run `load_dataset.py` for loading our Accident Detection Dataset into numpy matrices
2. Now Run `train.py` and then `test.py` for training and testing of Accident Dataset
3. Our Model is trained Now and we're ready to test our test_input for this run `frame_cut.py` 
4. And at last run `predict.py` for the detection of Accident and Non Accident footages for the given input

<br/>

### Pre-Trained Model<a name="preTrainedModel"></a>
The Pre-trained Model can be found here Pre-trained Model. Extract the zip file into the 'master folder' of your model. And, finally run 'test.py' to obtaining accuracy on 'test set'.


<br/>

### Contributors
1. [Piyush Kumar](https://github.com/prithvipandit)
2. [Ghanshyam Gupta](https://github.com/captain10x)


<br/>

### References<a name="ref"></a>

* [CADP Dataset](https://ankitshah009.github.io/accident_forecasting_traffic_camera)
* [Detrac Dataset](https://detrac-db.rit.albany.edu/)
* Manually Created Dataset

<br/>
          
          
  
          

 
