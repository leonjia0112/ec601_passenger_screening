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

# from threat_zone_predicting_runnable import input_pipeline
# from preprocess_one_image import preprocess_data
# from networks import vggnet

INPUT_FOLDER = 'input_data/'
PROCESSED_FOLDER = 'processed_data/'
STAGE1_LABELS = 'stage1_labels.csv'
TRAIN_PATH = 'train/'
MODEL_PATH = 'model/'

MODEL_NAME = 'model_tz_1'
PROCESSED_TZ = 'input-tz1-250-250.npy'

# overall_label = [0, 0]
overall_prediction = [0, 0, 0, 0]
THREAT_ZONE = 1

IMAGE_DIM = 250
LEARNING_RATE = 1e-3

#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses one image file
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------
def preprocess_data(infile):
    images = tsa.read_data(INPUT_FOLDER + infile)
    subject = infile.split('.')[0]  

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    images = images.transpose()
    threat_zone_examples = []
    # for each threat zone, loop through each image, mask off the zone and then crop it
    for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list, tsa.zone_crop_list)):

        threat_zone = threat_zone_x_crop_dims[0]
        crop_dims = threat_zone_x_crop_dims[1]

        # get label
        label = np.array(tsa.get_subject_zone_label(tz_num, tsa.get_subject_labels(STAGE1_LABELS, subject)))

        for img_num, img in enumerate(images):
            
            if threat_zone[img_num] is not None:
 
                base_img = np.flipud(img)
                rescaled_img = tsa.convert_to_grayscale(base_img)
                high_contrast_img = tsa.spread_spectrum(rescaled_img)
                masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                normalized_img = tsa.normalize(cropped_img)
                zero_centered_img = tsa.zero_center(normalized_img)
                threat_zone_examples.append([[tz_num], zero_centered_img, label])
    for tz_num, tz in enumerate(tsa.zone_slice_list):

        tz_examples_to_save = []
        tz_examples = [example for example in threat_zone_examples if example[0] == [tz_num]]
        tz_examples_to_save.append([[features_label[1], features_label[2]] for features_label in tz_examples])

        #save batch
        np.save(PROCESSED_FOLDER + 'input-tz{}-{}-{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1])), tz_examples_to_save)

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
  
def alexnet(width, height, lr):
    tf.reset_default_graph()
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
    
    # print(input_image)

    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    model.load(MODEL_PATH + MODEL_NAME)

    count = 0;
    fault = 0;
    for subject in os.listdir(INPUT_FOLDER):
        predict_image(subject, model)

        # print("+-----------------------+")
        # print("total: ", count)
        # print("fault: ", fault)
        # print("fault alarm ratio: ", fault/count*100)
        # print("+-----------------------+")

    # # final 
    # print("+FINAL------------------+")
    # print("total: ", count)
    # print("fault: ", fault)
    # print("fault alarm ratio: ", fault/count*100)
    # print("+-----------------------+")


def predict_image(input_file, model):
    preprocess_data(input_file)
    preccessed_image, label = input_pipeline(PROCESSED_TZ, PROCESSED_FOLDER)
    preccessed_image = preccessed_image.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

    result = model.predict(preccessed_image)
    right = 0
    length = len(result)
    for _, r in result:
        right += r

    # label [1 0] no threat
    if(label[0][0] > label[0][1]):

        # predict threat
        # [NO YES] ZONE 2
        if(right/length > 0.2):
            overall_prediction[1] += 1

        # predict not threat
        # [NO NO] ZONE 4 
        else:
            overall_prediction[3] += 1

    # label [0 1] has threat
    else:
        # predict threat
        # [YES YES] ZONE 1
        if(right/length > 0.2):
            overall_prediction[0] += 1

        # predict not threat
        # [YES NO] ZONE 3
        else:
            overall_prediction[2] += 1
    

    print("+-----------------------+")
    print('{} {}\n'.format(overall_prediction[0], overall_prediction[1]))
    print('{} {}\n'.format(overall_prediction[2], overall_prediction[3]))
    print("+-----------------------+\n")

if __name__ == '__main__':
    main()