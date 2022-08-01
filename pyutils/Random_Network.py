import numpy as np
from sklearn.preprocessing import StandardScaler

class RNN:
        def __init__(self,num_samples,num_attributes, num_hidden_neurons, mean=0,sd=0.1,
                     regularization=10e-4):
            self.mean=mean
            self.sd = sd
            self.samples = num_samples
            self.attributes = num_attributes
            self.num_hidden_neurons=num_hidden_neurons
            
            
            #set lambda value for regularization
            self.regularization=regularization
        
        #to obtain the hidden weights, we can use the gauss distribution or, based in 
        #(RIBAS,2020) we can also use the LCG method
        def set_hidden_weights(self,method='normal'):
            #get the hiddden weights
            if(method=='normal'):    
                self.hidden_weights = np.random.normal(self.mean, 
                                                       self.sd, 
                                                       size=[self.num_hidden_neurons,
                                                             self.attributes+1])
            elif(method=='lcg'):
                E=self.num_hidden_neurons*(self.attributes+1)
                a=E+2
                b=E+3
                c=E**2
                V = np.zeros(shape=self.num_hidden_neurons*(self.attributes+1))
                V[0] = E+1
                for i in range(1,len(V)):
                    V[i] = (a*V[i-1]+b)%c
                  
                self.hidden_weights = V.reshape(self.num_hidden_neurons, self.attributes+1)
                
            #compute z-score normalization of hidden weights
            std_scaler = StandardScaler()
            
            self.hidden_weights = std_scaler.fit_transform(self.hidden_weights)
            return self.hidden_weights
                    
            
        
        def get_output_weights(self, X, y):
            
            
            #transpose X matrix in order to multiply for H
            X_t = X.transpose()

            #add bias in matrix X
            bias = np.ones(shape=self.samples)
            X_t = np.vstack((bias*-1, X_t))

            #get Z matrix
            self.Z = np.tanh(np.matmul(self.hidden_weights, X_t))

            #add bias in matrx Z
            self.Z = np.vstack((bias*-1, self.Z))

            #get transpose of Z
            Z_t = np.transpose(self.Z)


            #multiply y for transpose of Z
            parcial_1 = np.matmul(y, Z_t)
            #multiply z for its transpose
            parcial_2 = np.matmul(self.Z, Z_t)

            #compute the Tikhonov regularization term
            reg = self.regularization*np.eye(parcial_2.shape[0])


            #calculate inverse of last computation
            parcial_3 = np.linalg.inv(parcial_2+reg)

            #compute f using the equation f = Y*Z_t*(Z*Z_t)^-1
            self.output_weights = np.matmul(parcial_1,parcial_3)

            return self.output_weights
        
        def predict_output(self, X, y):
            y_pred = np.matmul(self.output_weights,self.Z)
            
            return y_pred