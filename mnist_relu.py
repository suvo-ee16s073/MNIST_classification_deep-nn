
"""

# classification task on mnist data using deep neural network

# There are three hiddenlayer with  1000, 500, 250 units. we use cross entrophy as loss function  

# Important: in this code i implemented ReLu activation function but I did not change the name of the Sigmoid funtion here (as same    code i implemented for sigmoid previously) ,please read Sigmoid as ReLu here.

# momentum, learning rate scheduling, regularization  implemented here.

# This code is using only numpy, scipy.io and matplotlib.pyplot  

# p.s: sorry for being clumsy

"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt




"""load data"""
test_img = loadmat('t_ip_im_vals.mat') 
test_img = test_img['t_ip_im_vals']
test_img = test_img.T
test_labels = loadmat('t_lbl.mat')
test_labels = test_labels['t_lbl']
test_output = np.zeros([10, 10000], dtype=int) #converting to one hot representation
for i in range(0, 9999):
    test_output[test_labels[i, 0], i] = 1    
test_output=test_output.T


train_img = loadmat('train_im.mat') 
train_img = train_img['inputValues']
train_img = train_img.T
labels = loadmat('labels.mat')
labels = labels['labels']
output = np.zeros([10,60000],dtype=int) #converting to one hot representation


for i in range(0, 59999):
    output[labels[i, 0],i] = 1

output = output.T

i = None
labels = None



""" function definations"""


# compute sigmoid nonlinearity
def sigmoid(x):
    y = x
    a = y.shape
    for i in range(0, a[1]):
        if y[0, i] >= 0:
           y[0, i] = y[0, i]
        else:
           y[0, i] = 0
    return y

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(x):
    y = x
    a = y.shape
    for i in range(0, a[1]):
        if y[0, i] >= 0:
           y[0, i] = 1
        else:
           y[0, i] = 0
    return y

def CrossEn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.nan_to_num(-y*np.log(a))

def softmax_layer_output(weight, layer_input):       #softmax layer output
    layer_input1 = np.dot(a = weight, b = layer_input)
    layer_input1 -= np.max(layer_input1)
    temp = np.exp(layer_input1)
    return temp/np.sum(temp)
    

minibatchsize = 64
alphas = [0.1]   #learning rate


""" Training """


for alpha in alphas:
    print("\nTraining With Alpha:" + str(alpha))
    

    # randomly initialize our weights with mean 0
    synapse_0 = np.random.normal(0,0.08,(784,1000)) 
    synapse_1 = np.random.normal(0,0.08,(1000,500)) #weight initialization
    synapse_2 = np.random.normal(0,0.08,(500,250)) 
    synapse_3 = np.random.normal(0,0.08,(250,10)) 
    errr = list()
    errr_test = list()
    alpha_h = alpha
    momentum0 = 0
    momentum1 = 0 #momentum initialization
    momentum2 = 0
    momentum3 = 0
    for j in range(3000):
        l3w = np.zeros((250, 10))
        l2w = np.zeros((500, 250))
        l1w = np.zeros((1000, 500))
        l0w = np.zeros((784, 1000))
        err = 0#np.zeros((1,1))
        
        for i in range(1, minibatchsize):
            minibatch = np.random.randint(low=0, high = 59999,size = 1)
            """ Feed forward through layers 0, 1,2,3 and 4"""
            layer_0 = train_img[minibatch,:]
            layer_1 = sigmoid(np.dot(layer_0,synapse_0))
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))
            layer_3 = sigmoid(np.dot(layer_2,synapse_2))
            layer_4 = softmax_layer_output(layer_3,synapse_3)
            
            
            layer_4_error = layer_4 - output[minibatch, :]
            err += np.linalg.norm(CrossEn(layer_4, output[minibatch, :]))
    
    
            '''backpropagarion'''
            layer_4_delta = layer_4_error
            l3w += layer_3.T.dot(layer_4_delta)
            layer_3_error = layer_4_delta.dot(synapse_3.T)
            layer_3_delta = layer_3_error * sigmoid_output_to_derivative(np.dot(layer_2, synapse_2))
            l2w += layer_2.T.dot(layer_3_delta)
            layer_2_error = layer_3_delta.dot(synapse_2.T)
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(np.dot(layer_1, synapse_1))
            l1w += layer_1.T.dot(layer_2_delta)
            layer_1_error = layer_2_delta.dot(synapse_1.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(np.dot(layer_0, synapse_0))
            l0w += layer_0.T.dot(layer_1_delta)
            
        
        """test error"""
        if (j% 100) == 0:
            y_hat = np.zeros((10000,1))
            err_test = 0
            for i in range(0,9999):
                layer_0_test = test_img[np.array([i]), :]
                layer_1_test = sigmoid(np.dot(layer_0_test, synapse_0))
                layer_2_test = sigmoid(np.dot(layer_1_test, synapse_1))
                layer_3_test = sigmoid(np.dot(layer_2_test, synapse_2))
                layer_4_test = softmax_layer_output(layer_3_test, synapse_3)
                err_test += np.linalg.norm(CrossEn(layer_4_test, test_output[i,:]))
            errr_test.append(err_test/10000)
        
        err = err/64
        errr.append(err)
        if (j% 100) == 0:
            print("Error after "+str(j)+" iterations:" + str(err))
        if (j%250) == 0:      #LR scheduling
            alpha_h = alpha_h*0.85
        """momentum"""    
        momentum0 = 0.9 * momentum0 - alpha_h * (l0w/64)
        momentum1 = 0.9 * momentum1 - alpha_h * (l1w/64)
        momentum2 = 0.9 * momentum2 - alpha_h * (l2w/64) #momentum update
        momentum3 = 0.9 * momentum3 - alpha_h * (l3w/64)  
        """weight update"""
        synapse_3 = (1 - alpha_h * 0.005) * synapse_3 + momentum3
        synapse_2 = (1 - alpha_h * 0.005) * synapse_2 + momentum2
        synapse_1 = (1 - alpha_h * 0.005) * synapse_1 + momentum1  #gradient decent 
        synapse_0 = (1 - alpha_h * 0.005) * synapse_0 + momentum0




""" test accuracy and error ploting """


y_hat = np.zeros((10000, 1))
for i in range(0, 9999):
    layer_0 = test_img[np.array([i]), :]
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))  
    layer_3 = sigmoid(np.dot(layer_2, synapse_2))
    layer_4 = softmax_layer_output(layer_3, synapse_3)
    temp = np.array(np.where(layer_4==layer_4.max()))
    y_hat[i,0] = temp[1, 0]
acc = test_labels-y_hat    
np.count_nonzero(acc)



train_epochs = np.arange(0, 3000)
test_epochs = np.arange(0, 3000,100)
  



# save loss figure
fig1=plt.figure(1)
plt.plot(train_epochs, errr,'r', label = 'Training_loss') 
plt.plot(test_epochs[0:30], errr_test,'b', label = 'Testing loss') 
fig_nam = "ReLU_with_alpha_" + str(alphas[0]) + "_Training_and_Testing_loss.eps" 
fig_name = "ReLU_with_alpha_" + str(alphas[0]) + "_Training_and_Testing_loss" 
plt.legend(loc='upper right')
plt.title(fig_name)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
#ax=plt.gca
fig1.savefig(fig_nam)




# randomly chosen 20 images to show top 3 predictions
idx_20=loadmat('c_20.mat') 
idx_20=idx_20['c_20']
 # top 3 predictions for 20 randomly choosen test images
t_20_arr = [];
for ii in range(0,20):
    layer_0 = test_img[idx_20[0][ii], :]
    layer_0.resize(1, 784)
    l1 = np.dot(layer_0, synapse_0)
    layer_1 = sigmoid(l1)
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    layer_3 = sigmoid(np.dot(layer_2, synapse_2))
    layer_4 = softmax_layer_output(layer_3, synapse_3)
    t20_tgt = test_labels[idx_20[0][ii], :][0]
    op_y = layer_4.copy()
    op_y1 =  layer_4.copy()
    op_y1.sort()
    tp1=np.where(op_y[0] == op_y1[0][-1])[0][0]
    tp2=np.where(op_y[0] == op_y1[0][-2])[0][0]
    tp3=np.where(op_y[0] == op_y1[0][-3])[0][0]

    tp = [idx_20[0][ii], t20_tgt, tp1, tp2, tp3]
    t_20_arr.append(tp)

print("top 3 predictions for 20 randomly choosen test images \n")
print(np.array(t_20_arr)) 


