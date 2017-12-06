
# coding: utf-8

# In[11]:


# import libraries
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
import random
from timeit import default_timer as timer
import tsahelper as tsa
# to confirm desired modules loaded
print('modules loaded')

#---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
INPUT_FOLDER = 'data'
PREPROCESSED_DATA_FOLDER = 'data/pps_augmented/'
STAGE1_LABELS = 'stage1_labels.csv'
## each images has 16 views. Each of 17 Zone is not visible in all 16 views.
# Zone 1 is visible in:- 11 Zone 2 is visible in:- 11 Zone 3 is visible in:- 12
# Zone 4 is visible in:- 12 Zone 5 is visible in:- 8 Zone 6 is visible in:- 9
# Zone 7 is visible in:- 8 Zone 8 is visible in:- 11 Zone 9 is visible in:- 9
# Zone 10 is visible in:- 8 Zone 11 is visible in:- 13 Zone 12 is visible in:- 14
# Zone 13 is visible in:- 13 Zone 14 is visible in:- 10 Zone 15 is visible in:- 12
# Zone 16 is visible in:- 13 Zone 17 is visible in:- 8 
# Sum of above all is 182
EXAMPLES_PER_SUBJECT = 182 # total number of zones visible in all views per image
BATCH_SIZE = 16
FILE_LIST = []
IMAGE_DIM = 250
# In[21]:


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------

def preprocess_tsa_data():

    # OPTION 1: get a list of all subjects for which there are labels
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    SUBJECT_LIST = df['Subject'].unique()

    # OPTION 2: get a list of all subjects for whom there is data
    # SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]
    #print(len(SUBJECT_LIST))
    #print(SUBJECT_LIST)

    # OPTION 3: get a list of subjects for small bore test purposes
    #SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
    #                '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
    #                '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e',
    #                '47e2a4a8e13ec7100f6af8cd839d1bb3','e087226320cc189142228b5fb93ed58f']

    # intialize tracking and saving items
    batch_num = 1
    count = 0
    threat_zone_examples = []
    start_time = timer()
    print(len(SUBJECT_LIST))
    for subject in SUBJECT_LIST:
        count += 1
        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time,
                                                                     subject))
        print('--------------------------------------------------------------')
        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list,
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num,
                             tsa.get_subject_labels(STAGE1_LABELS, subject)))
           # print(STAGE1_LABELS, subject)
            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))
                if label[0] == 0:
                    print('threat is present')
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
                        print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                           high_contrast_img.mean()))

                        # get the masked image
                        print('-> masking image')
                        masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                        print('-> shape {}|mean={}'.format(masked_img.shape,
                                                           masked_img.mean()))

                        # crop the image
                        print('-> cropping image')
                        cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                        print('-> shape {}|mean={}'.format(cropped_img.shape,
                                                           cropped_img.mean()))

                        # normalize the image
                        print('-> normalizing image')
                        normalized_img = tsa.normalize(cropped_img)
                        print('-> shape {}|mean={}'.format(normalized_img.shape,
                                                           normalized_img.mean()))

                        # zero center the image
                        print('-> zero centering')
                        zero_centered_img = tsa.zero_center(normalized_img)
                        print('-> shape {}|mean={}'.format(zero_centered_img.shape,
                                                           zero_centered_img.mean()))

                        # append the features and labels to this threat zone's example array
                        print ('-> appending example to threat zone {}'.format(tz_num))
                        threat_zone_examples.append([[tz_num], zero_centered_img, label])
                        center = (125,125)
                        M = cv2.getRotationMatrix2D(center, 5, 1.0)
                        rotated = cv2.warpAffine(zero_centered_img, M, (250, 250))
                        # print('rotated image shape {} | mean= {}'.format(zero_centered_img.shape,
                                                                        #  zero_centered_img.mean()))
                        # cv2.imwrite("thumbnail.png", cropped)
                        # cv2.imwrite("rotated.jpg", rotated)
                        # cv2.imshow("original.jpg", zero_centered_img)
                        # cv2.waitKey(0)
                        # cv2.imshow("rotated.jpg", rotated)
                        # cv2.waitKey(0)
                        threat_zone_examples.append([[tz_num], rotated, label])
                        M = cv2.getRotationMatrix2D(center, 10, 1.0)
                        rotated1 = cv2.warpAffine(zero_centered_img, M, (250, 250))
                        threat_zone_examples.append([[tz_num], rotated1, label])
                        # cv2.imshow("rotated1.jpg", rotated1)
                        # cv2.waitKey(0)
                        M = cv2.getRotationMatrix2D(center, 15, 1.0)
                        rotated2 = cv2.warpAffine(zero_centered_img, M, (250, 250))
                        threat_zone_examples.append([[tz_num], rotated2, label])
                        # cv2.imshow("rotated2.jpg", rotated2)
                        # cv2.waitKey(0)
                        # M = cv2.getRotationMatrix2D(center, 20, 1.0)
                        rotated3 = cv2.warpAffine(zero_centered_img, M, (250, 250))
                        threat_zone_examples.append([[tz_num], rotated3, label])
                        # cv2.imshow("rotated3.jpg", rotated3)
                        # cv2.waitKey(0)
                        print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                                             len(threat_zone_examples),
                                                             len(threat_zone_examples[0]),
                                                             len(threat_zone_examples[0][0]),
                                                             len(threat_zone_examples[0][1][0]),
                                                             len(threat_zone_examples[0][1][1]),
                                                             len(threat_zone_examples[0][2])))
                    else:
                        print('-> No view of tz:{} in img:{}. Skipping to next...'.format(
                                    tz_num, img_num))
                    print('------------------------------------------------')
                else:
                    print('threat not present and label is', label[0])
                    if count >= 0:
                        # count = 0
                        print('IN LOOP')
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
                            print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                               high_contrast_img.mean()))

                            # get the masked image
                            print('-> masking image')
                            masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                            print('-> shape {}|mean={}'.format(masked_img.shape,
                                                               masked_img.mean()))

                            # crop the image
                            print('-> cropping image')
                            cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                            print('-> shape {}|mean={}'.format(cropped_img.shape,
                                                               cropped_img.mean()))

                            # normalize the image
                            print('-> normalizing image')
                            normalized_img = tsa.normalize(cropped_img)
                            print('-> shape {}|mean={}'.format(normalized_img.shape,
                                                               normalized_img.mean()))

                            # zero center the image
                            print('-> zero centering')
                            zero_centered_img = tsa.zero_center(normalized_img)
                            print('-> shape {}|mean={}'.format(zero_centered_img.shape,
                                                               zero_centered_img.mean()))

                            # append the features and labels to this threat zone's example array
                            print ('-> appending example to threat zone {}'.format(tz_num))
                            threat_zone_examples.append([[tz_num], zero_centered_img, label])
                            print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                                                 len(threat_zone_examples),
                                                                 len(threat_zone_examples[0]),
                                                                 len(threat_zone_examples[0][0]),
                                                                 len(threat_zone_examples[0][1][0]),
                                                                 len(threat_zone_examples[0][1][1]),
                                                                 len(threat_zone_examples[0][2])))
                        # count = 0

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
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER
                    + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1,
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]),
                      batch_num))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] ==
                           [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]]
                                        for features_label in tz_examples])

            #save batch
            np.save(PREPROCESSED_DATA_FOLDER +
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1,
                                                     len(threat_zone_examples[0][1][0]),
                                                     len(threat_zone_examples[0][1][1]),
                                                     batch_num),
                                                     tz_examples_to_save)
            # print('tz examples:-',tz_examples)
# # unit test ---------------------------------------
preprocess_tsa_data()

