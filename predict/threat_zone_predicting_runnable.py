# import libraries
from __future__ import print_function
from __future__ import division
import tsahelper as tsa
import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer



#---------------------------------------------------------------------------------------
# Constants
#
# INPUT_FOLDER:                 The folder that contains the source data
#
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files 
# 
# STAGE1_LABELS:                The CSV file containing the labels by subject
#
# THREAT_ZONE:                  Threat Zone to train on (actual number not 0 based)
#
# BATCH_SIZE:                   Number of Subjects per batch
#
# EXAMPLES_PER_SUBJECT          Number of examples generated per subject
#
# FILE_LIST:                    A list of the preprocessed .npy files to batch
# 
# TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train and test
#
# TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training
#
# TEST_SET_FILE_LIST:           The list of .npy files to be used for testing
#
# IMAGE_DIM:                    The height and width of the images in pixels
#
# LEARNING_RATE                 Learning rate for the neural network
#
# N_TRAIN_STEPS                 The number of train steps (epochs) to run
#
# TRAIN_PATH                    Place to store the tensorboard logs
#
# MODEL_PATH                    Path where model files are stored
#
# MODEL_NAME                    Name of the model files
#
#----------------------------------------------------------------------------------------
INPUT_FOLDER = 'image_file/'
PROCESSED_FOLDER = 'processed_image/'
STAGE1_LABELS = 'stage1_labels.csv'

THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'train/'
MODEL_PATH = 'model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE )) 



#---------------------------------------------------------------------------------------
# input_pipeline(filename, path): prepares a batch of features and labels for training
#
# parameters:      filename - the file to be batched into the model
#                  path - the folder where filename resides
#
# returns:         feature_batch - a batch of features to train or test on
#                  label_batch - a batch of labels related to the feature_batch
#
#---------------------------------------------------------------------------------------
def input_pipeline(filename, path):
    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []
    
    #Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))
        
    #Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)
    
    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])
    
    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)
    
    return feature_batch, label_batch
  

#---------------------------------------------------------------------------------------
# alexnet(width, height, lr): defines the alexnet
#
# parameters:      width - width of the input image
#                  height - height of the input image
#                  lr - learning rate
#
# returns:         none
#
#-------------------------------------------------------------------------------------
def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model


def main():
    input_image, input_image_label = input_pipeline('input-tz1-250-250.npy', PROCESSED_FOLDER)
    input_image = input_image.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

    # print(input_image)

    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    model.load(MODEL_PATH + MODEL_NAME + '-173')
    result = model.predict(input_image)
    left = 0
    right = 0
    for l, r in result:
        left += l
        right += r
    print('Image label')
    print(input_image_label)
    print('Predict result')
    print(result)    
    print('-------------------------------')
    if(left > right):
        print('There is not thread for zone 1!')
    else:
        print('There is thread for zone 1!')
    print('-------------------------------')


if __name__ == '__main__':
    main()