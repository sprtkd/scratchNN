# scratchNN
This is an example of a self-made fully connected neural network from scratch in Python (only numpy used)
The neural network is trained on MNIST dataset.

This was made as a part of summer internship at CrowdyLabs
http://crowdylab.droppages.com/

# You can download the following files and keep them in the same directory as your python shell:
https://pjreddie.com/media/files/mnist_test.csv
https://pjreddie.com/media/files/mnist_train.csv
# Or
# Extract the zip file: 
    mnist_dataset.zip

The main training and testing datasets are in csv format.

The images consist of 784 pixels and there are 10 distinct labels of digits 0-9.

The current neural network consists of the following layers:

•	1 input layer consisting of 784 input nodes.

•	3 hidden layers consisting of 500, 300, 100 nodes respectively.

•	1 output layer consisting of 10 output nodes.

The activation function used in the hidden nodes is a rectified linear unit (ReLU).

The output of the output layer was passed through a softmax function.

Regularization has been used.

The size of training set used is 60000 and the test set size used is 10000.

The maximum accuracy achieved is 86%.

# To train, run basicNNtrain.py

The trained neural network model is saved as data.pickle

# To predict, run basicNNpredict.py
    and give the path of the image you want to predict
# Samples images are in the testing_img.zip file. 
   Please extract it.

A sample output during training is given as sample_output_tr.txt

A sample ouput for prediction is given as sample_output_te.txt

The accuracy is little low, if anyone finds a bug, please report.

# Created by Suprotik Dey
  supdey1@gmail.com
