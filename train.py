import numpy as np
from scipy.signal import convolve2d

def initialize_weights(path="default"):
    weights = []
    if path != "default":
        #load weights
        print('hello')
    else:
        
        weights.append(np.zeros((16,3,3)))
        weights.append(np.zeros((16,3,3)))
        weights.append(np.zeros((32,6400)))
        weights.append(np.zeros((10,32)))

    return weights

def relu(x):
    activate = lambda x: x if x>=0 else 0
    relufunc = np.vectorize(activate)
    return relufunc(x)
    
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def conv2d(x,filters):
    output = np.zeros((filters.shape[0],x.shape[0],x.shape[1]))
    if len(x.shape)>2:
        for f in filters:
            for i,ch in enumerate(x):
                output[i]+=convolve2d(ch,f)
                
    return output



def run_network(x,weights):
    #architecture
    output = relu(conv2d(x,weights[0]))
    output = relu(conv2d(x,weights[1]))
    output = output.flatten()
    print(output.shape)
    output = relu(np.dot(weights[2],output))
    output = softmax(np.dot(weights[3],output))
    
    return output
         
def d_relu(x):
    activate = lambda x: 1 if x>=0 else 0
    relufunc = np.vectorize(activate)
    return relufunc(x)       

def d_softmax(x):
        
    
    
def upgrade_network(x,y,weights):
    print('todo')
    #conv1
    x0 = conv2d(x,weights[0])
    y0 = relu(x0)
    s0 = d_relu(x0)
    
    #conv2
    x1 = conv2d(y0,weights[1])
    y1 = relu(x1)
    s1 = d_relu(x1)
    
    #dense
    x2 = np.dot(weights[2],y1)
    y2 = relu(x2)
    s2 = d_relu(x2)
    
    #softmax
    x3 = np.dot(weights[2],y2)
    y3 = relu(x3)
    s3 = d_relu(x3)
    
    d3 = y3- y
    g3 = np.dot(d3,s3)
    
    
def save_weights():
    print('todo')

def train():
    print('todo')


if __name__ =='__main__':
    print('Training network')
    test = np.ones((2,3,3))
    test2 = - test

    print(relu(test))
    print(relu(test2))
    
    test_img = np.zeros((20,20))
    
    weights = initialize_weights()
    for i in range(3200):
        result = run_network(test_img,weights)
    
    print(result)





