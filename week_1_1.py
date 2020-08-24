# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:52:15 2020

@author: boels
"""

import pandas as pd
import numpy as np
import seaborn as sns
import util
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2 
import os
from PIL import Image





train_list_files_benign, train_list_files_cancerous, test_list_files_benign, \
        test_list_files_cancerous = util.read_filenames()


#remarques: train_list_files_benign has 7 augmented files per raw file (7/8).
train_b = util.not_augmented_files(train_list_files_benign)

#remarques: train_list_files_cancerous has 1 augmented files per raw file (1/2).
train_c = util.not_augmented_files(train_list_files_cancerous)

#remarques: test_list_files_benign has 7 augmented files per raw file (7/8).
test_b = util.not_augmented_files(test_list_files_benign)

#remarques: test_list_files_cancerous has 1 augmented files per raw file (1/2).
test_c = util.not_augmented_files(test_list_files_cancerous)


# list o file list.
filelists = [train_b, train_c, test_b, test_c]

# create dataframes
df_trb, df_trc, df_tb, df_tc = util.create_dataframes(filelists, train_b, train_c, test_b, test_c)



print(f'There are {len(df_trb)} unique Patient IDs in the training negative set')
print(f'There are {len(df_trc)} unique Patient IDs in the training positive set')
print(f'There are {len(df_tb)} unique Patient IDs in the test negative set')
print(f'There are {len(df_tc)} unique Patient IDs in the test positive set')


# 80 / 20 split.
# with 50% positives in 20% test set.
# with 50% positives in validation set (from the training set).
# and no overlap between train and test set.

# positives in training = 377 > negatives in training = 94.
# positives in test = 127 > negatives in test = 33.

# sum = 377 + 127 = 504 positive labels.

# 504 x 50% = 257 positive labels.

# total number of unique images = 94 + 377 + 33 + 127 = 631
# 631 x 80% = 504 images for the trainig set.

# test set:
# 631 x 20% = 127 images for the test set.
# test set = 127 / 2 = 64 images of positives labels
# and 63 images of negative labels.

# validation set:
# 10% from the 504 train images = 50 images.
# 50 / 2 = 25 images of positive labels.

# training set:
# 631 x 80% = 504 images for the trainig set.
# remaining positives labels = 504 - 64 - 25 = 415
# 504 - 415 = 89 

# merge and shuffle all data frames, create datasets based on labels.



df_train, df_test = util.create_datasets(df_trb, df_trc, df_tb, df_tc)

df_train = util.drop_overlapping_patients_train(df_train, df_test)


# insert images in train data frame
train_path = 'Data/raw train/'
test_path = 'Data/raw test/'

df_train = util.df_add_arrays(train_path, df_train)
df_test = util.df_add_arrays(test_path, df_test)



# Data Exploration
util.plot_label_frequency(df_train, df_test)


# Data Visualization
## Visualize a random selection of images from the train dataset.

# local_folder = 'Data/raw train/'
# file_name = 'demd159_CC_Right.tif'

# # use functions
# util.show_tif(local_folder, df_train)
# util.show_cv2(local_folder, file_name)


# Plot a histogram of the distribution of the pixels
util.plot_pixel_values(df_train)



# Data Preprocessing and Data Augmentation.

print(f'The data type of the pixels for each image is: {df_train.array[0].dtype}')

# normalize images and save as png files
util.add_png_arrays(df_train)
util.add_png_arrays(df_test)

util.add_png_filenames(df_train)
util.add_png_filenames(df_train)

## DATA VISUALIZATION

# Extract numpy values from Image column in data frame
images = df_train['images_png'].values

# Extract 9 random images from it
random_images = [np.random.choice(images) for i in range(9)]

# Location of the image dir
img_dir = 'Data/all/'
print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(20,20))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()




## INVESTIGATE A SIGNLE IMAGE

# Get the first image that was listed in the train_df dataframe
sample_img = df_train.images[4]
raw_image = plt.imread(os.path.join(img_dir, sample_img))
plt.imshow(raw_image, cmap='gray')
plt.colorbar()
plt.title('Raw breast X Ray Image')
print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, with {raw_image.shape[2]} color channels")
print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")


## INVESTIGATE PIXEL VALUE DISTRIBUTION.


# Plot a histogram of the distribution of the pixels
raw_image = df_train.array[4]
sns.distplot(raw_image.ravel(), 
             label=f'Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')


'------------------------------TEST: Process_input from Imagenet Generator-------------------------'
## IMAGE PREPROCESSING IN KERAS


IMAGE_DIR = "Data/out/train/"
target_w = 320
target_h = 320
BATCH_SIZE_TRAINING = 10
BATCH_SIZE_VALIDATION = 5
BATCH_SIZE_TESTING = 1

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input

data_generator = ImageDataGenerator(horizontal_flip = True,
                                    vertical_flip = True,
                                    preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_dataframe(
            dataframe=df_train,
            directory=IMAGE_DIR,
            x_col='images_png',
            y_col=['malignancy'],
            class_mode="raw",
            batch_size=BATCH_SIZE_TRAINING,
            shuffle=True,
            seed=1,
            target_size=(target_w,target_h))

train_generator_keras, label = train_generator.__getitem__(4)


'-------------------------------------------------------------------------------'

generator_center_norm = util.get_train_generator(df_train, 'Data/out/train/', 'images_png', \
                                'malignancy', shuffle=True, batch_size=8, seed=1, \
                                target_w = 320, target_h = 320,
                                samplewise_center= True, samplewise_std_normalization= False,
                                rescale= 1/255.0
                                )

generator_normaliz = util.get_train_generator(df_train, 'Data/out/train/', 'images_png', \
                                'malignancy', shuffle=True, batch_size=8, seed=1, \
                                target_w = 320, target_h = 320,
                                samplewise_center= False, samplewise_std_normalization= False,
                                rescale= 1/255.0)
'--------------------------------------------------------------------------------'
# Show a processed image
sns.set_style("white")
raw_image, label = generator_normaliz.__getitem__(4)


plt.imshow(raw_image[0], cmap='gray')
plt.title('Raw Breast X Ray Image')
# print(f"The dimensions of the image are {generated_image_norm.shape[1]} pixels width and {generated_image_norm.shape[2]} pixels height")
# print(f"The maximum pixel value is {generated_image_norm.max():.4f} and the minimum is {generated_image_norm.min():.4f}")
# print(f"The mean value of the pixels is {generated_image_norm.mean():.4f} and the standard deviation is {generated_image_norm.std():.4f}")



## HISTOGRAM TO COMPARE RAW AND PREPROCESSED IMAGES

generator_cent_norm, label = generator_center_norm.__getitem__(4)
generator_norm, label = generator_normaliz.__getitem__(4)

# Include a histogram of the distribution of the pixels
sns.set()
plt.figure(figsize=(10, 7))

# Plot histogram for generated image
sns.distplot(generator_cent_norm[0].ravel(), 
             label=f'Generated Image: mean {np.mean(generator_cent_norm[0]):.4f} - Standard Deviation {np.std(generator_cent_norm[0]):.4f} \n'
             f'Min pixel value {np.min(generator_cent_norm[0]):.4} - Max pixel value {np.max(generator_cent_norm[0]):.4}', 
             color='red', 
             kde=False)

sns.distplot(train_generator_keras[2].ravel(), 
             label=f'Original Image: mean {np.mean(generator_norm):.4f} - Standard Deviation {np.std(generator_norm):.4f} \n '
             f'Min pixel value {np.min(generator_norm):.4} - Max pixel value {np.max(generator_norm):.4}',
             color='blue', 
             kde=False)

# Place legends
plt.legend(loc='best')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixel')

