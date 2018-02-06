#!/usr/bin/env python

from zipfile import ZipFile
import numpy as np

class DataLoader:
    def __init__(self):
        path = "../data_Ass23/"
        pass

    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'

        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
            m = labels.shape[0]
            labels_onehot = np.zeros((m,10))
            labels_onehot[np.arange(m),labels] = 1

        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
            
        # print images.shape, labels.shape
        return images.T,labels_onehot.T

    def create_batches(self,X,y,nbatch,seed,batch_sz=128):
        permutation = list(np.random.permutation(X.shape[1]))
        shuffle_X = X[:,permutation]
        shuffle_y = y[:,permutation]
        minibatches = [(shuffle_X[:,k*batch_sz:(k+1)*batch_sz],shuffle_y[:,k*batch_sz:(k+1)*batch_sz]) for k in range(nbatch)]
        return minibatches

    # def create_batches(self,X,y,nbatch,seed,batch_sz = 128):
    #     randomize = np.arange(X.shape[0])
    #     np.random.shuffle(randomize)
    #     shuffle_X = X[randomize,:]
    #     shuffle_y = y[randomize,:]
    #     minibatches = [(shuffle_X[k*batch_sz:(k+1)*batch_sz,:],shuffle_y[k*batch_sz:(k+1)*batch_sz,:]) for k in range(nbatch)]
    #     return minibatches



# DL = DataLoader()
# DL.load_data(mode = 'test')
