import numpy as np
from sklearn.externals import joblib
import random


class Batcher:
    def __init__(self,storage,data,batch_size,context_length,id2vec):
        self.context_length = context_length
        self.storage = storage
        self.data = data
        self.num_of_samples = int(data.shape[0])
        self.dim = 300 #len(id2vec[0])
        self.num_of_labels = data.shape[1] - 4  - 70 
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(self.num_of_samples / self.batch_size) 
        self.id2vec = id2vec
        self.pad  = np.zeros(self.dim)
        self.pad[0] = 1.0
        
    
    def create_input_output(self,row):
        s_start = row[0]
        s_end = row[1]
        e_start = row[2]
        e_end = row[3]
        labels = row[74:]
        features = row[4:74]
        seq_context = np.zeros((self.context_length*2 + 1,self.dim))        
        temp = [ self.id2vec[self.storage[i]][:self.dim] for i in range(e_start,e_end)]
        mean_target = np.mean(temp,axis=0)
        
        j = max(0,self.context_length - (e_start - s_start))
        for i in range(max(s_start,e_start - self.context_length),e_start):
            seq_context[j,:] = self.id2vec[self.storage[i]][:self.dim]
            j += 1
        seq_context[j,:] = np.ones_like(self.pad)
        j += 1
        for i in range(e_end,min(e_end+self.context_length,s_end)):
            seq_context[j,:] = self.id2vec[self.storage[i]][:self.dim]
            j += 1

        return seq_context, mean_target, labels, features
        

    def next(self):
        X_context = np.zeros((self.batch_size,self.context_length*2+1,self.dim))
        X_target_mean = np.zeros((self.batch_size,self.dim)) 
        Y = np.zeros((self.batch_size,self.num_of_labels))
        F = np.zeros((self.batch_size,70),np.int32)
        for i in range(self.batch_size):
            X_context[i,:,:], X_target_mean[i,:], Y[i,:], F[i,:] = self.create_input_output(self.data[self.batch_num * self.batch_size + i,:])
        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return [X_context, X_target_mean, Y, F] 
                                        
    def shuffle(self):
        np.random.shuffle(self.data)



    

    
