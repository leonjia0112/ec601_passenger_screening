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

import tsahelper as tsa

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
INPUT_FOLDER = 'aps/'
PREPROCESSED_DATA_FOLDER = 'tsa/'
STAGE1_LABELS = 'stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'tsa_logs/train/'
MODEL_PATH = 'tsa_logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE ))



#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------

def preprocess_tsa_data():
    
    # OPTION 1: get a list of all subjects for which there are labels
    # df = pd.read_csv(STAGE1_LABELS)
    # df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    # SUBJECT_LIST = df['Subject'].unique()

    # OPTION 2: get a list of all subjects for whom there is data
    SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]
    
    # OPTION 3: get a list of subjects for small bore test purposes
    # SUBJECT_LIST = ['0043db5e8c819bffc15261b1f1ac5e42',
    # 	'00360f79fd6e02781457eda48f85da90',
    #     '01c08047f617de893bef104fb309203a',
    #     '011516ab0eca7cad7f5257672ddde70e',
    #     '01941f33fd090ae5df8c95992c027862',
    #     '0050492f92e22eed3474ae3a6fc907fa',
    #     '0097503ee9fa0606559c56458b281a08']
    
    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()
    
    for subject in SUBJECT_LIST:

        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, 
                                                                     subject))
        print('--------------------------------------------------------------')
        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list, tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num, tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))
                
                if threat_zone[img_num] is not None:

                    # correct the orientation of the image
                    print('-> reorienting base image') 
                    base_img = np.flipud(img)
                    print('-> shape {}|mean={}'.format(base_img.shape, 
                                                       base_img.mean()))

                    # convert to grayscale
                    print('-> converting to grayscale')
                    rescaled_img = tsa.convert_to_grayscale(base_img)
                    print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                       rescaled_img.mean()))

                    # spread the spectrum to improve contrast
                    print('-> spreading spectrum')
                    high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape,high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')
                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    print('-> shape {}|mean={}'.format(masked_img.shape, masked_img.mean()))

                    # crop the image
                    print('-> cropping image')
                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                    print('-> shape {}|mean={}'.format(cropped_img.shape, cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')
                    normalized_img = tsa.normalize(cropped_img)
                    print('-> shape {}|mean={}'.format(normalized_img.shape, normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')
                    zero_centered_img = tsa.zero_center(normalized_img)
                    print('-> shape {}|mean={}'.format(zero_centered_img.shape,zero_centered_img.mean()))

                    # append the features and labels to this threat zone's example array
                    print ('-> appending example to threat zone {}'.format(tz_num))
                    threat_zone_examples.append([[tz_num], zero_centered_img, label])
                    print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(len(threat_zone_examples),len(threat_zone_examples[0]),len(threat_zone_examples[0][0]),len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1]),len(threat_zone_examples[0][2])))
                else:
                    print('-> No view of tz:{} in img:{}. Skipping to next...'.format(tz_num, img_num))
                print('------------------------------------------------')

        # each subject gets EXAMPLES_PER_SUBJECT number of examples (182 to be exact, 
        # so this section just writes out the the data once there is a full minibatch 
        # complete.
        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 
                                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format( 
                                        tz_num+1,
                                        len(threat_zone_examples[0][1][0]),
                                        len(threat_zone_examples[0][1][1]), 
                                        batch_num))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] == 
                               [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} is a 
                # tz_num 1 based in the minibatch file to select which batches to 
                # use for training a given threat zone
                np.save(PREPROCESSED_DATA_FOLDER + 
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]), 
                                                         batch_num), 
                                                         tz_examples_to_save)
                del tz_examples_to_save

            #reset for next batch 
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1
    

    # we may run out of subjects before we finish a batch, so we write out 
    # the last batch stub
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]), len(threat_zone_examples[0][1][1]), batch_num))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] == [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]] for features_label in tz_examples])

            #save batch
            np.save(PREPROCESSED_DATA_FOLDER + 
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]), len(threat_zone_examples[0][1][1]), batch_num), tz_examples_to_save)
# unit test ---------------------------------------
preprocess_tsa_data() 