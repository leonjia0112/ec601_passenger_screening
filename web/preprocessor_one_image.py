# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re
import sys

import random
from timeit import default_timer as timer

import tsahelper as tsa

#---------------------------------------------------------------------------------------
# Constants
#
# INPUT_FOLDER:                 The folder that contains the source data
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files 
# STAGE1_LABELS:                The CSV file containing the labels by subject
#----------------------------------------------------------------------------------------
INPUT_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed_image/'
STAGE1_LABELS = 'stage1_labels.csv'


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses one image file
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------
def preprocess_tsa_data(infile):
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
            # print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
            # print('Threat Zone Label -> {}'.format(label))
            
            if threat_zone[img_num] is not None:

                # correct the orientation of the image
                # print('-> reorienting base image') 
                base_img = np.flipud(img)
                # print('-> shape {}|mean={}'.format(base_img.shape, base_img.mean()))

                # convert to grayscale
                # print('-> converting to grayscale')
                rescaled_img = tsa.convert_to_grayscale(base_img)
                # print('-> shape {}|mean={}'.format(rescaled_img.shape, rescaled_img.mean()))

                # spread the spectrum to improve contrast
                # print('-> spreading spectrum')
                high_contrast_img = tsa.spread_spectrum(rescaled_img)
                # print('-> shape {}|mean={}'.format(high_contrast_img.shape,high_contrast_img.mean()))

                # get the masked image
                # print('-> masking image')
                masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                # print('-> shape {}|mean={}'.format(masked_img.shape, masked_img.mean()))

                # crop the image
                # print('-> cropping image')
                cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                # print('-> shape {}|mean={}'.format(cropped_img.shape, cropped_img.mean()))

                # normalize the image
                # print('-> normalizing image')
                normalized_img = tsa.normalize(cropped_img)
                # print('-> shape {}|mean={}'.format(normalized_img.shape, normalized_img.mean()))

                # zero center the image
                # print('-> zero centering')
                zero_centered_img = tsa.zero_center(normalized_img)
                # print('-> shape {}|mean={}'.format(zero_centered_img.shape,zero_centered_img.mean()))

                # append the features and labels to this threat zone's example array
                # print ('-> appending example to threat zone {}'.format(tz_num))
                threat_zone_examples.append([[tz_num], zero_centered_img, label])
                print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(len(threat_zone_examples),len(threat_zone_examples[0]),len(threat_zone_examples[0][0]),len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1]),len(threat_zone_examples[0][2])))
            else:
                print('-> No view of tz:{} in img:{}. Skipping to next...'.format(tz_num, img_num))
            print('------------------------------------------------')


    for tz_num, tz in enumerate(tsa.zone_slice_list):

        tz_examples_to_save = []

        # write out the batch and reset
        print(' -> writing: ' + PROCESSED_FOLDER + 'preprocessed_TSA_scans-tz{}-{}-{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1])))

        # get this tz's examples
        tz_examples = [example for example in threat_zone_examples if example[0] == [tz_num]]

        # drop unused columns
        tz_examples_to_save.append([[features_label[1], features_label[2]] for features_label in tz_examples])

        #save batch
        np.save(PROCESSED_FOLDER + 'input-tz{}-{}-{}.npy'.format(tz_num+1, len(threat_zone_examples[0][1][0]),len(threat_zone_examples[0][1][1])), tz_examples_to_save)


# process data if input exist
def main():
    if(len(sys.argv) > 1):
        file = sys.argv[1]
        # file = '826b3b5eb25ddd6f7d2aed1e531e69b9.aps'
        preprocess_tsa_data(file) 

if __name__ == "__main__":
    main()



