import numpy as np
import csv
import pickle
import matplotlib as plt
##flow--
##input data from mnist csv -> input layer -> hidden layers -> output layer
##there are several neurons in each layer each of which contains an array of weights and a bias
##then it will pass through an activation function

##input data as 1 hot encoding labels
##output as probabilities
##using mnist dataset
##28x28 pixels -> 784 inputs
##ouputs -> 10 digits 0-9


#data loader
def csv2labeldata(stringname):
    '''This function loads csv as labeled data and returns labels and data as columns
    please give string name as input. csv will be split from 2nd column'''
    with open(stringname, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        x = list(reader)
        result = np.array(x).astype("float")
        sh = result.shape
        labels,data=np.hsplit(result, [1])
        print("labels",labels)
        print("data",data)
        return labels,data

def onehotencoder(labels,start,stop):
    dist = 1 - start + stop
    size=labels.shape[0]
    mat = np.zeros((size,dist))
    for i in range(size):
        mat[i,int(labels[i])] = 1.0
    return mat
    

def mnistloader():
    '''return in format train_labels,train_data,test_labels,test_data'''
    try:
        print("Reading..Please wait")
        ltest,dtest = csv2labeldata("mnist_test.csv")
        ltest2=onehotencoder(ltest,0,9)
        print("Reading..Please wait")
        ltrain,dtrain = csv2labeldata("mnist_train.csv")
        ltrain2=onehotencoder(ltrain,0,9)  
                
        print("Done.")
        return ltrain2,dtrain,ltest2,dtest
    except:
        print("Load failed!")
        

#mnist load
train_labels, train_data, test_labels, test_data = mnistloader()
print('train_labels shape:', train_data.shape)
#definite randomize init
np.random.seed(8)


#size of neural network
hidden_layer_num = 2
training_size = 60000
test_size = 10000

input_nodes = train_data.shape[1]
hidden1_nodes = 512
hidden2_nodes = 512
output_nodes = 10

#hyperparameters
learning_rate = 0.00001
train_epoch_per_data = 20
reg_term = 0.01

#activation function
def relu(x):
    y=x
    for i in range(x.shape[0]):
        y[i]=np.maximum(0,x[i])
    return y   
   
def relu_derivative(x):
    y=x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(x[i][j]>0.0):
                y[i][j]= 1.0
            else:
                y[i][j]= 0.0
    return y
   

def softmax(x):
    exparr = np.exp(x-np.max(x))
    f = exparr/exparr.sum(axis=1)
    return f

#cross entropy
def crossEntropy(output,labels):
    return (np.dot(labels,np.log(output)))

#definintion of nn
#each layer will have a matrix of weights and an array of biases
    
#hidden layer 1
hid_layer1_weights = np.random.randn(input_nodes, hidden1_nodes)
hid_layer1_weights = hid_layer1_weights/(np.amax(hid_layer1_weights))
hid_layer1_biases = np.zeros((1,hidden1_nodes))

#hidden layer 2
hid_layer2_weights = np.random.randn(hidden1_nodes ,hidden2_nodes)
hid_layer2_weights = hid_layer2_weights/(np.amax(hid_layer2_weights))
hid_layer2_biases = np.zeros((1,hidden2_nodes))

#output layer
output_layer_weights = np.random.randn(hidden2_nodes ,output_nodes)
output_layer_weights = output_layer_weights/(np.amax(output_layer_weights))
output_layer_biases = np.zeros((1,output_nodes))

#actual model param tuple
model_params = (hid_layer1_weights,hid_layer1_biases, hid_layer2_weights,hid_layer2_biases, output_layer_weights,output_layer_biases)


def loadmodel(path):
    with open(path, 'rb') as model_file:
        data = pickle.load(model_file)

    return data

def savemodel(path,model_pars):
    print("\nWriting model to file...")
    with open(path, 'wb') as model_file:
        pickle.dump(model_pars, model_file)

    print("\nDone")



def feedforward(input_data,model_par):
    hlw_1,hlb_1, hlw_2,hlb_2, ow,ob = model_par
    m1 = np.dot(input_data ,hlw_1) + hlb_1
    z1 = relu(m1)
    m2 = np.dot(z1 ,hlw_2) + hlb_2
    z2 = relu(m2)
    m3 = np.dot(z2, ow) + ob
    predicted_prob=softmax(m3)
    return (predicted_prob,z1,z2)




def backprop(input_data,input_label,transfer_model,model_par,epsilon):
    predicted_prob,z1,z2 = transfer_model
    hlw_1,hlb_1, hlw_2,hlb_2, ow,ob = model_par
    
    #compute the derivative of the weights and biases and return the updated ones
    #d stands for derivative

    delta3 = (predicted_prob - input_label)  #predict - label
    dow = np.dot(z2.T,delta3)
    dob = np.sum(delta3)

    delta2 = np.multiply(np.dot(delta3,ow.T) , relu_derivative(z2))
    
    dhlw_2 =np.dot(z1.T,delta2) 
    dhlb_2 =np.sum(delta2)

    delta1 =np.multiply(np.dot(delta2,hlw_2.T) , relu_derivative(z1))
    
    dhlw_1 =np.dot(input_data.T,delta1) 
    dhlb_1 =np.sum(delta1)


    #regularization
    dhlw_1 += reg_term * hlw_1
    dhlw_2 += reg_term * hlw_2
    dow += reg_term * ow

    #updating
    hlw_1 -= epsilon*dhlw_1
    hlb_1 -= epsilon*dhlb_1
    hlw_2 -= epsilon*dhlw_2
    hlb_2 -= epsilon*dhlb_2
    ow -= epsilon*dow
    ob -= epsilon*dob
    
    return (hlw_1,hlb_1, hlw_2,hlb_2, ow,ob)




def train_nn(input_data,input_label,model_par,epsilon):
    #feedforward
    transfer_model_params=feedforward(input_data,model_par)   
    
    #backpropagation
    
    model_par=backprop(input_data,input_label,transfer_model_params,model_par,epsilon)
    
    return (model_par,transfer_model_params[0])

def predict_mnist(input_data,model_pars):
    x=feedforward(input_data,model_pars)
    return x[0]



    
def accuracy_calc(t_data,t_labels,model_pars,t_size):
    summation=0
    for i in range(t_size):
        if(np.argmax(predict_mnist(t_data[i],model_pars)) == np.argmax(t_labels[i])):
            summation+=1
    return ((100.0 * summation)/t_size)

def accuracy_calc_cross_entropy(t_data,t_labels,model_pars,t_size):
    summation=0
    for i in range(t_size):
        summation+=crossEntropy(predict_mnist(t_data[i],model_pars),t_labels[i])
    return summation

def accuracy_calc_mse(t_data,t_labels,model_pars,t_size):
    summation=0
    for i in range(t_size):
        summation+=((predict_mnist(t_data[i],model_pars) - t_labels[i])** 2.0).mean(axis=1)
    return ((100.0 * summation)/t_size)


#actual Loop
print("\nTraining started")
for j in range(training_size):
    for i in range(train_epoch_per_data):
        #train nn for single epoch
        tr_data = train_data[j].reshape((1,train_data[j].shape[0]))
        model_params,pred = train_nn(tr_data,train_labels[j],model_params,learning_rate)
        tr_label = train_labels[j].reshape((1,train_labels[j].shape[0]))
        if (accuracy_calc_mse(tr_data,tr_label,model_params,1)==0.0):
            break        
    #calculate loss and print at regular intervals
    if j%1000==0:
        print("\n-------------------------------------------------")
        print("\nData number:",j)
        print("\nPredicted_probabilities",pred)
        print("\nActual Label",train_labels[j])
        print('\nTraining accuracy for current:',accuracy_calc(tr_data,tr_label,model_params,1),"%")
        print('\nTest accuracy:',accuracy_calc(test_data,test_labels,model_params,test_size),"%")
        print('\nTraining error:',accuracy_calc_mse(tr_data,tr_label,model_params,1))
        print('\nTest error:',accuracy_calc_mse(test_data,test_labels,model_params,test_size))
        print(((training_size-j-1)*100/(training_size)),"% training left..")
print("\nTraining finished!!")

print('\nTraining accuracy:',accuracy_calc(train_data,train_labels,model_params,training_size),"%")
print('\nTest accuracy:',accuracy_calc(test_data,test_labels,model_params,test_size),"%")

savemodel('data.pickle',model_params)


