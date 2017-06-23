#this code helps to test prediction from an image of a digit

import numpy as np

#for handling files and images
import pickle
from PIL import Image

def relu(x):
    y=x
    for i in range(x.shape[0]):
        y[i]=np.maximum(0,x[i])
    return y
   
def softmax(x):
    exparr = np.exp(x-np.max(x))
    f = exparr/exparr.sum(axis=1)
    return f

def loadmodel(path):
    with open(path, 'rb') as model_file:
        data = pickle.load(model_file)

    return data

def feedforward(input_data,model_par):
    hlw_1,hlb_1, hlw_2,hlb_2, hlw_3,hlb_3, ow,ob = model_par
    m1 = np.dot(input_data ,hlw_1) + hlb_1
    z1 = relu(m1)
    m2 = np.dot(z1 ,hlw_2) + hlb_2
    z2 = relu(m2)
    m3 = np.dot(z2, hlw_3) + hlb_3
    z3 = relu(m3)
    m4 = np.dot(z3 ,ow) + ob
    predicted_prob=softmax(m4)
    
    return (predicted_prob,z1,z2,z3)

def predict_mnist_digit(input_data,model_pars):
    x=feedforward(input_data,model_pars)
    return x[0]


model_params = loadmodel('data.pickle')
imgpath = input("\nEnter name of image:\n")

img = Image.open(imgpath).convert('L')
img_as_np = np.asarray(img)

#data is to be sent as 1, 784
#reshape the image for processing
width, height = img_as_np.shape
img_to_predict = np.reshape(img_as_np,(width*height))

print("\nThe predicted output is",np.argmax(predict_mnist_digit(img_to_predict,model_params)))


