import numpy as np
from scipy.signal import convolve2d
from sklearn.datasets import load_digits


def load_mnist_data():
    data = load_digits(return_X_y=True)
    x_train = np.zeros((1500,8,8))
    y_train = np.zeros((1500,10))
    
    for i,im in enumerate(data[0][:1500]):
        x_train[i] = im.reshape((8,8))
        y_train[i][data[1][i]] = 1

    x_test = np.zeros((data[0].shape[0]-1500,8,8))
    y_test = np.zeros((data[0].shape[0]-1500,10))
    for i,im in enumerate(data[0][1500:]):
        x_test[i] = im.reshape((8,8))
        y_test[i][data[1][i+1500]] = 1

    return x_train,y_train,x_test,y_test
    
def initialize_weights(path="default"):
    weights = []
    if path != "default":
        #load weights
        print('hello')
    else:
        alpha0 = 0.05
        weights.append(alpha0*np.random.sample((16,3,3))-alpha0/2)
        weights.append(alpha0*np.random.sample((16,3,3))-alpha0/2)
        weights.append([alpha0*np.random.sample((1024,32))-0.5*alpha0,alpha0*np.random.sample(32)-0.5*alpha0])
        weights.append([alpha0*np.random.sample((32,10))-0.5*alpha0,alpha0*np.random.sample(10)-0.5*alpha0])

    return weights

def relu(x):
    activate = lambda x: x if x>=0 else -0.01*x
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
    #print('run network')
    output = relu(conv2d(x,weights[0]))
    #print(output.shape)
    output = relu(conv2d(output,weights[1]))
    #print(output.shape)
    output = output.flatten()
    #print(output.shape)
    output = relu(np.dot(output,weights[2][0])+weights[2][1])
    #print(output.shape)
    output = softmax(np.dot(output,weights[3][0])+weights[3][1])
    #print(output.shape)
    return output
         
def d_relu(x):
    activate = lambda x: 1 if x>=0 else -0.01
    relufunc = np.vectorize(activate)
    return relufunc(x)       

def evaluate(data,labels,weights):
    ct_error=0.
    for x,y in zip(data,labels):
        x_pred = np.argmax(run_network(x,weights))
        if x_pred==np.argmax(y):
            ct_error+=1.
    print('Evaluation accuracy: '+str(ct_error/len(data)))
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
                    for i in range(y.shape[0]-fshape[1]):
                        for j in range(y.shape[1]-fshape[2]):
                            delta[c,a,b] +=d[c,i,j] *y[i+a,j+b]
                    
    return delta
def cross_entropy(y_true,y_predicted):
    return -1.*np.sum(y_true*np.log(y_predicted))
        
def upgrade_network(x_batch,y_batch,weights,lrate):
    #print(x_batch.shape)
    delta0 = np.zeros(weights[0].shape)
    delta1 = np.zeros(weights[1].shape)
    delta2 = [np.zeros(weights[2][0].shape),np.zeros(weights[2][1].shape)]
    delta3 = [np.zeros(weights[3][0].shape),np.zeros(weights[3][1].shape)]
    LOSS=0.
    #print('todo')
    for x,y in zip(x_batch,y_batch):
        #conv1
        x0 = conv2d(x,weights[0])
        y0 = relu(x0)
        #print('c1 '+str(np.max(y0)))
        s0 = d_relu(x0)
        
        #conv2
        x1 = conv2d(y0,weights[1])
        y1 = relu(x1)
        #print('c2 '+str(np.max(y1)))
        s1 = d_relu(x1)
        
        y1_flatten = y1.flatten()
        
        #dense
        x2 = np.dot(y1_flatten,weights[2][0])+weights[2][1]
        y2 = relu(x2)
        s2 = d_relu(x2)
        #print('d1 '+str(np.max(y2)))
        
        #softmax
        x3 = np.dot(y2,weights[3][0])+weights[3][1]
        y3 = softmax(x3)
        s3 = d_relu(x3)
        #print('d2 '+str(np.max(y3)))
        #print('weights')
        #for i in range(4):
            #print(np.max(weights[i]))
        LOSS += cross_entropy(y,y3)
        
        
        #back propagation
        d3 = y-y3
        #print(d3)
        delta3[0] +=  np.dot(np.array([y2]).T,np.array([d3]))
        delta3[1]+=d3
        gamma2 = np.dot(d3,weights[3][0].T)
        
        d2 = s2 * gamma2
        delta2[1]+=d2
        delta2[0] += np.dot(np.array([y1_flatten]).T,np.array([d2]))
        #print('shape '+str(delta2.shape))
        gamma1_flatten = np.dot(d2,weights[2][0].T)
        
        #unflatten    
        gamma1 = gamma1_flatten.reshape(y1.shape)
             
        d1 = s1*gamma1
    
        delta1 += compute_delta(d1,y0,weights[1].shape)
        gamma0 = np.sum([conv2d(d1[c],flip_filter(weights[1][c])) for c in range(weights[1].shape[0])])
        
        d0 = s0*gamma0
        delta0 += compute_delta(d0,x,weights[0].shape)
        #print(np.max(delta0))
      
        #print(np.max(delta3))
    LOSS/=len(x_batch)
    #print('LOSS >> '+str(LOSS))
    delta0/=len(x_batch)
    delta1/=len(x_batch)
    delta2[0]/=len(x_batch)
    delta2[1]/=len(x_batch)
    delta3[0]/=len(x_batch)
    delta3[1]/=len(x_batch)
    
    weights[0]+=lrate*delta0
    weights[1]+=lrate*delta1    
    weights[2][0]+=lrate*delta2[0]
    weights[2][1]+=lrate*delta2[1]
    weights[3][0]+=lrate*delta3[0]
    weights[3][1]+=lrate*delta3[1]
    #print(np.max(delta3[0]),np.max(delta0),np.max(delta1),np.max(delta2[0]))
    return weights,LOSS 

    
    
def save_weights(weights,path):
    np.savez_compressed(path+'w0.npz',weights[0])
    np.savez_compressed(path+'w1.npz',weights[1])
    np.savez_compressed(path+'w20.npz',weights[2][0])
    np.savez_compressed(path+'w21.npz',weights[2][1])
    np.savez_compressed(path+'w30.npz',weights[3][0])
    np.savez_compressed(path+'w31.npz',weights[3][1])
    
    print('Saving done')

        

def train():
    print('todo')


if __name__ =='__main__':
    print('Training network')
    test = np.ones((2,3,3))
    test2 = - test

    print(relu(test))
    print(relu(test2))
    
    test_img = np.zeros((8,8))
    
    weights = initialize_weights()
    for i in range(1):
        result = run_network(test_img,weights)
    
    print(result)
    
    y = np.zeros(10)
    y[2] = 1

    print('SHAPES')
    print(test_img.shape)
    print(y.shape)
    
    #weights = upgrade_network(test_img,y,weights,lrate =0.001)
    
    x_train,y_train,x_test,y_test = load_mnist_data()
    
    weights = initialize_weights()
    
    lrate =0.1
    epochs = 60
    decay = lrate/epochs
    

    for e in range(epochs):
        print('<>'*10)
        print('epoch '+ str(e))
        for i in range(46):
            #print('BATCH '+str(i))
            weights,LOSS = upgrade_network(x_train[32*i:32*(i+1)],y_train[32*i:32*(i+1)],weights,lrate=lrate)
            if LOSS>100:
                print('Loss explosion')
                break
        print('LOSS >> '+str(LOSS))
        if i%10==0:
            save_weights(weights,'models/')
        evaluate(x_test,y_test,weights)
        lrate-=decay

