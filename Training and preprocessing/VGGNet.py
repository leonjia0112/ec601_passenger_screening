# coding: utf-8

# This file contains the coding for another neural network called VGGNet.
"""we tested with this net and turns out this is painfully slow for this dataset.
VGG Net has 16 layes and most of them are convolution 
"""


# import libraries
from __future__ import print_function
from __future__ import division

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
print('modules loaded')
import tsahelper as tsa


# Get the assiged number of cores for this job. This is stored in
# the NSLOTS variable, If NSLOTS is not defined throw an exception.
def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')


# In[20]:


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
#INPUT_FOLDER = 'data'
PREPROCESSED_DATA_FOLDER = 'data/pps/'
#STAGE1_LABELS = 'data/stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 64
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.25
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'Logs/train/'
MODEL_PATH = 'Logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM,
                                                IMAGE_DIM, THREAT_ZONE ))

def get_train_test_file_list():

    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print ('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER)
                     if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        train_test_split = len(FILE_LIST) -                            max(int(len(FILE_LIST)*TRAIN_TEST_SPLIT_RATIO),1)
        #FILE_LIST=random.shuffle(FILE_LIST)
        #print(FILE_LIST)
        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        print('Train/Test Split -> {} file(s) of {} used for testing'.format(
              len(FILE_LIST) - train_test_split, len(FILE_LIST)))

# unit test ----------------------------
#get_train_test_file_list()
print(len(FILE_LIST))
#print (


# In[23]:


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

# unit test ------------------------------------------------------------------------
print ('Train Set -----------------------------')
for f_in in TRAIN_SET_FILE_LIST:
    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
    print (' -> features shape {}:{}:{}'.format(len(feature_batch),
                                                len(feature_batch[0]),
                                                len(feature_batch[0][0])))
    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))

print ('Test Set -----------------------------')
for f_in in TEST_SET_FILE_LIST:
    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
    print (' -> features shape {}:{}:{}'.format(len(feature_batch),
                                                len(feature_batch[0]),
                                                len(feature_batch[0][0])))
    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))


# In[24]:


#---------------------------------------------------------------------------------------
# shuffle_train_set(): shuffle the list of batch files so that each train step
#                      receives them in a different order since the TRAIN_SET_FILE_LIST
#                      is a global
#
# parameters:      train_set - the file listing to be shuffled
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list

# Unit test ---------------
#print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)
shuffle_train_set(TRAIN_SET_FILE_LIST)
#print ('After Shuffling ->', TRAIN_SET_FILE_LIST)


# In[25]:


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

def VGGNet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.001, name='labels')
    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model

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
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model
# In[ ]:


#---------------------------------------------------------------------------------------
# train_conv_net(): runs the train op
#
# parameters:      none
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def train_conv_net():

    val_features = []
    val_labels = []
#    feature_batch=[]
#    label_batch=[]

    # get train and test batches
    get_train_test_file_list()

    # instantiate model
    model = VGGNet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)

    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in,
                                                                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
    print('validation batch shape ->', val_features.shape)

#Below commented part is to load the data by batch size(64x250x20) instead of dumping the full data (which is ~17000x250x250).
#To run this code in PC it is better to uncomment 324,327,330,332,336,340-343 lines and comment lines 348-360.

# start training process
#    for i in range(N_TRAIN_STEPS):

        # shuffle the train set files before each step
#        shuffle_train_set(TRAIN_SET_FILE_LIST)

        # run through every batch in the training set
#        for  f_in in TRAIN_SET_FILE_LIST:
            # read in a batch of features and labels for training
#            feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
            #print('tmp feature batch:-',tmpfeature_batch.shape())
            #feature_batch=np.concatenate((tmpfeature_batch,feature_batch),axis=0)
            #label_batch=np.concatenate((tmplabel_batch,label_batch),axis=0)
#            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            #print ('Feature Batch Shape ->', feature_batch.shape)

            # run the fit operation
#            model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1,
#                      validation_set=({'features': val_features}, {'labels': val_labels}),
#                      shuffle=True, snapshot_step=20, show_metric=True,
#                      run_id=MODEL_NAME)


    X = []
    Y = []    
    for i,all_f_in in enumerate(FILE_LIST):
        if i == 0:
	    X, Y = input_pipeline(all_f_in, PREPROCESSED_DATA_FOLDER)
	else:
	    X_temp, Y_temp = input_pipeline(all_f_in, PREPROCESSED_DATA_FOLDER)
            X = np.concatenate((X_temp, X),axis=0)
	    Y = np.concatenate((Y_temp, Y),axis=0)
    X = X.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
    print('X shape:-', X.shape)
    model.fit({'features':X}, {'labels':Y}, n_epoch=100, validation_set = 0.2, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=100,
          snapshot_epoch=False, run_id='VGG_Net')	 

# unit test -----------------------------------
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=get_n_cores()-1,
      inter_op_parallelism_threads=1,
      allow_soft_placement=True, 
      log_device_placement=True)

sess = tf.Session(config=session_conf)
tflearn.is_training(True, session=sess)
train_conv_net()
#testing the image
# for f_in in TEST_SET_FILE_LIST:
#     predictions = model.predict(f_in)
#     print(predictions)
