# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:34:17 2017

@author: Antoine
"""
import numpy as np
from scipy.signal import convolve2d
from sklearn.datasets import load_digits

#function to load MNIST dataset
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
    
#############################################################
#Main class
#
#This class allows to train and use a small convolutional network
class small_cnn:
    
    def __init__(self):
        self.weights = []   
    
    #derivative of RELU activation function
    def _relu(self,x):
        activate = lambda x: x if x>=0 else -0.01*x
        relufunc = np.vectorize(activate)
        return relufunc(x)
        
    #Softmax final activation function
    def _softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x))
    
    
    #Discrete convolution
    def _conv2d(self,x,filters):
        
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
  
    #Derivative of RELU activation function             
    def _d_relu(self,x):
        activate = lambda x: 1 if x>=0 else -0.01
        relufunc = np.vectorize(activate)
        return relufunc(x)       
    
    #Computes weight update for convolutional multi channels layers
    def _compute_delta(self,d,y,fshape):
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
    
    #Loss function
    def _cross_entropy(self,y_true,y_predicted):
        return -1.*np.sum(y_true*np.log(y_predicted)) 

    #Array double flip
    def _flip_filter(self,w):
        return np.flip(np.flip(w,0),1)
    
    #Upgrades weights using examples (x_batch,y_batch), thanks to gradient descent
    def _upgrade_network(self,x_batch,y_batch,weights,lrate):
        
        #deltas initialization
        delta0 = np.zeros(weights[0].shape)
        delta1 = np.zeros(weights[1].shape)
        delta2 = [np.zeros(weights[2][0].shape),np.zeros(weights[2][1].shape)]
        delta3 = [np.zeros(weights[3][0].shape),np.zeros(weights[3][1].shape)]
        LOSS=0.
        
        #batch gradient computations
        for x,y in zip(x_batch,y_batch):
            
            #conv1
            x0 = self._conv2d(x,weights[0])
            y0 = self._relu(x0)
            s0 = self._d_relu(x0)
            
            #conv2
            x1 = self._conv2d(y0,weights[1])
            y1 = self._relu(x1)
            s1 = self._d_relu(x1)
            
            y1_flatten = y1.flatten()
            
            #dense
            x2 = np.dot(y1_flatten,weights[2][0])+weights[2][1]
            y2 = self._relu(x2)
            s2 = self._d_relu(x2)
            
            #softmax
            x3 = np.dot(y2,weights[3][0])+weights[3][1]
            y3 = self._softmax(x3)

            LOSS += self._cross_entropy(y,y3)
            
            
            #back propagation
            
            d3 = y-y3
            delta3[0] +=  np.dot(np.array([y2]).T,np.array([d3]))
            delta3[1]+=d3
            gamma2 = np.dot(d3,weights[3][0].T)
            
            d2 = s2 * gamma2
            delta2[1]+=d2
            delta2[0] += np.dot(np.array([y1_flatten]).T,np.array([d2]))            
            gamma1_flatten = np.dot(d2,weights[2][0].T)
            
            #unflatten    
            gamma1 = gamma1_flatten.reshape(y1.shape)
                 
            d1 = s1*gamma1        
            delta1 += self._compute_delta(d1,y0,weights[1].shape)
            gamma0 = np.sum([self._conv2d(d1[c],self._flip_filter(weights[1][c])) for c in range(weights[1].shape[0])])
            
            d0 = s0*gamma0
            delta0 += self._compute_delta(d0,x,weights[0].shape)
            
        #mean over batch    
        LOSS/=len(x_batch)
        
        delta0/=len(x_batch)
        delta1/=len(x_batch)
        delta2[0]/=len(x_batch)
        delta2[1]/=len(x_batch)
        delta3[0]/=len(x_batch)
        delta3[1]/=len(x_batch)
        
        #weight update
        weights[0]+=lrate*delta0
        weights[1]+=lrate*delta1    
        weights[2][0]+=lrate*delta2[0]
        weights[2][1]+=lrate*delta2[1]
        weights[3][0]+=lrate*delta3[0]
        weights[3][1]+=lrate*delta3[1]

        return weights,LOSS    

       

    def initialize_weights(self):
        alpha0 = 0.05
        self.weights.append(alpha0*np.random.sample((16,3,3))-alpha0/2)
        self.weights.append(alpha0*np.random.sample((16,3,3))-alpha0/2)
        self.weights.append([alpha0*np.random.sample((1024,32))-0.5*alpha0,alpha0*np.random.sample(32)-0.5*alpha0])
        self.weights.append([alpha0*np.random.sample((32,10))-0.5*alpha0,alpha0*np.random.sample(10)-0.5*alpha0])

    def train(self,data,labels,test_data,test_labels,initial_lrate=0.05,epochs = 50):
        
        n_batch = int(len(data)/32)
        lrate = initial_lrate
        decay = initial_lrate/epochs
        
        for e in range(epochs):
            print('<>'*10)
            print('epoch '+ str(e)+'/'+str(epochs))
            for i in range(n_batch):
                self.weights,LOSS =self._upgrade_network(data[32*i:32*(i+1)],labels[32*i:32*(i+1)],self.weights,lrate=lrate)
                if LOSS>100:
                    print('Loss explosion')
                    break
            print('LOSS >> '+str(LOSS))
            if e%10==0:
                self.save_weights('models/autosave-ep'+str(e))
            self.evaluate(test_data,test_labels)
            lrate-=decay
        


    def predict(self,x):
        #architecture
        #conv1
        output = self._relu(self._conv2d(x,self.weights[0]))
        
        #conv2
        output = self._relu(self._conv2d(output,self.weights[1]))
        
        #flatten
        output = output.flatten()
        
        #fully-connected
        output = self._relu(np.dot(output,self.weights[2][0])+self.weights[2][1])
        
        #softmax layer
        output = self._softmax(np.dot(output,self.weights[3][0])+self.weights[3][1])
        
        return output        
    
    #Computes average accuracy on the data
    def evaluate(self,data,labels):
        ct_error=0.
        for x,y in zip(data,labels):
            x_pred = np.argmax(self.predict(x))
            if x_pred==np.argmax(y):
                ct_error+=1.
        print('Evaluation accuracy: '+str(ct_error/len(data)))

    #save weights into npz files
    def save_weights(self,path):
        np.savez_compressed(path+'w0.npz',self.weights[0])
        np.savez_compressed(path+'w1.npz',self.weights[1])
        np.savez_compressed(path+'w20.npz',self.weights[2][0])
        np.savez_compressed(path+'w21.npz',self.weights[2][1])
        np.savez_compressed(path+'w30.npz',self.weights[3][0])
        np.savez_compressed(path+'w31.npz',self.weights[3][1])
        
        print('Saving done')
        
if __name__=='__main__':
    

    

    
    data_train,label_train,data_test,label_test = load_mnist_data()
    
    model = small_cnn()
    model.initialize_weights()
    model.train(data_train,label_train,data_test,label_test,initial_lrate=0.1,epochs=60)
    model.save_weights('weights/model-')
    
    
    