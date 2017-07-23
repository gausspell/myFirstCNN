import numpy as np
from scipy.signal import convolve2d

def initialize_weights(path="default"):
    weights = []
    if path != "default":
        #load weights
        print('hello')
    else:
        alpha0 = 0.001
        weights.append(alpha0*np.ones((16,3,3)))
        weights.append(alpha0*np.ones((16,3,3)))
        weights.append(alpha0*np.ones((32,6400)))
        weights.append(alpha0*np.ones((10,32)))

    return weights

def relu(x):
    activate = lambda x: x if x>=0 else 0
    relufunc = np.vectorize(activate)
    return relufunc(x)
    
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def conv2d(x,filters):
    
    if len(x.shape)>2:
        output = np.zeros((filters.shape[0],x.shape[1],x.shape[2]))
        for i,f in enumerate(filters):
            for j in range(x.shape[0]):
                output[i]+=convolve2d(x[j],f,mode='same')
    else:
        if len(filters.shape)>2:
            output = np.zeros((filters.shape[0],x.shape[0],x.shape[1]))
            for i,f in enumerate(filters):      
                output[i] = convolve2d(x,f,mode='same')
        else:
            output = convolve2d(x,filters,mode='same')
                
    return output



def run_network(x,weights):
    #architecture
    print('run network')
    output = relu(conv2d(x,weights[0]))
    print(output.shape)
    output = relu(conv2d(output,weights[1]))
    print(output.shape)
    output = output.flatten()
    print(output.shape)
    output = relu(np.dot(weights[2],output))
    print(output.shape)
    output = softmax(np.dot(weights[3],output))
    print(output.shape)
    return output
         
def d_relu(x):
    activate = lambda x: 1 if x>=0 else 0
    relufunc = np.vectorize(activate)
    return relufunc(x)       

#def d_softmax(x):
    
def flip_filter(w):
    return np.flip(np.flip(w,0),1)

def compute_delta(d,y,fshape):
    delta = np.zeros((16,3,3))
    for c in range(fshape[0]):
        for a in range(fshape[1]):
            for b in range(fshape[2]):
                if len(y.shape)>2:
                    for k in range(y.shape[0]):
                        for i in range(y.shape[1]-fshape[1]):
                            for j in range(y.shape[2]-fshape[2]):
                                delta[c,a,b] +=d[c,i,j] *y[k,i+a,j+b]
                else:
                    for i in range(y.shape[0]-fshape[0]):
                        for j in range(y.shape[1]-fshape[1]):
                            delta[c,a,b] +=d[c,i,j] *y[i+a,j+b]
                    
    return delta
    
def upgrade_network(x,y,weights,lrate):
    print('todo')
    #conv1
    x0 = conv2d(x,weights[0])
    y0 = relu(x0)
    s0 = d_relu(x0)
    
    #conv2
    x1 = conv2d(y0,weights[1])
    y1 = relu(x1)
    s1 = d_relu(x1)
    
    y1_flatten = y1.flatten()
    
    #dense
    x2 = np.dot(weights[2],y1_flatten)
    y2 = relu(x2)
    s2 = d_relu(x2)
    
    #softmax
    x3 = np.dot(weights[3],y2)
    y3 = relu(x3)
    s3 = d_relu(x3)
    
    #back propagation
    d3 = y3 - y
    delta3 = np.dot(y3.T,d3)
    gamma2 = np.dot(y2.T,delta3)
    
    d2 = s2 * gamma2
    delta2 = np.dot(y2.T,d2)
    gamma1_flatten = np.dot(y1.T,delta2)
    
    #unflatten    
    gamma1 = gamma1_flatten.reshape(y1.shape)
         
    d1 = s1*gamma1
    for k in range(weights[0].shape[0]):
        for c in range(weights[1].shape[0]):
            print(k,c)
            print(d1[0,:d1.shape[1]-weights[1].shape[1]+1,:d1.shape[2]-weights[1].shape[2]+1].shape)
            print(y0[0].shape)
            print('<>'*10)
    delta1 = compute_delta(d1,y0,weights[1].shape)
    gamma0 = np.sum([conv2d(d1[c],flip_filter(weights[1][c])) for c in range(weights[1].shape[0])])
    
    d0 = s0*gamma0
    delta0 = compute_delta(d0,x,weights[0].shape)
  
    print(delta0.shape)
    weights[0]+=lrate*delta0
    weights[1]+=lrate*delta1    
    weights[2]+=lrate*delta2    
    weights[3]+=lrate*delta3

    return weights 

    
    
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
    for i in range(1):
        result = run_network(test_img,weights)
    
    print(result)
    
    y = np.zeros(10)
    y[2] = 1

    print('SHAPES')
    print(test_img.shape)
    print(y.shape)
    
    weights = upgrade_network(test_img,y,weights,lrate =0.001)



