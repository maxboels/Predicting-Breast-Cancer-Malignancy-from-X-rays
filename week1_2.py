# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:33:50 2020

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

from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K


## IMPORT DATA


# read filenames
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

# add dataset to dataframes
df_train, df_test = util.create_datasets(df_trb, df_trc, df_tb, df_tc)

# drop overlapping patients 
df_train = util.drop_overlapping_patients_train(df_train, df_test)

# insert images in train data frame
train_path = 'Data/raw train/'
test_path = 'Data/raw test/'

# add arrays
df_train = util.df_add_arrays(train_path, df_train)
df_test = util.df_add_arrays(test_path, df_test)

# rescale images to [0, 255] and save as png files
util.add_png_filenames(df_train)
util.add_png_filenames(df_test)

# create validation set with 50% of positives.
df_val, df_train = util.create_df_val(df_train)



## DATA VISUALIZATION


# # train 0 and 1
# util.show_png_images(df_train, 0, 'Data/out/train/')


## CLASS IMBALANCE


# WEIGHTED LOSS FUNCTION

# compute the class frequency
freq_pos_train, freq_neg_train = util.compute_label_freq(df_train)
freq_pos_val, freq_neg_val = util.compute_label_freq(df_val)
freq_pos_test, freq_neg_test = util.compute_label_freq(df_test)

# Visualization class frequency
sns.set(style="whitegrid")
data = pd.DataFrame({"Dataset": ['Train', 'Train', 'Val', 'Val', 'Test', 'Test'],
                     "Label": ['Negative','Positive','Negative','Positive','Negative','Positive'], 
                     "Frequency": [freq_neg_train, freq_pos_train,
                                    freq_neg_val, freq_pos_val,
                                    freq_neg_test, freq_pos_test]})
f = sns.barplot(x="Dataset", y="Frequency", hue="Label" ,data=data)
plt.title('Imbalanced Training set')



# calculate the positive weight as the fraction of negative labels
pos_weights = freq_neg_train
# calculate the negative weight as the fraction of positive labels
neg_weights = freq_pos_train

'________METHOD 1:___________OVERSAMPLING______________________________________'

## OVERSAMPLING CLASS IMBALANCE
#df_train = util.oversampling(df_train)

'______METHOD 2 : WEIGHTED LOSS FUNCTION_______________________________'

# see util.py
'______METHOD 3:_____WEIGHTED CLASS______________________________________'

# get class weights for fit()
class_weight = util.get_class_weights(df_train)



## DATA AUGMENTATION


from keras.preprocessing.image import ImageDataGenerator

'----------------------choose the pre-processing generator-------------------'

#from keras.applications.resnet50 import preprocess_input

from keras.applications.resnet_v2 import preprocess_input

#from keras.applications.densenet import preprocess_input

#from keras.applications.vgg19 import preprocess_input

'-----------------------------------------------------------------------'

IMAGE_DIR = "Data/out/train/"
target_w = 320
target_h = 320
BATCH_SIZE_TRAINING = 10
BATCH_SIZE_VALIDATION = 5
BATCH_SIZE_TESTING = 1

'------------------------------TEST: Process_input from Imagenet Generator-------------------------'
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

valid_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        directory= 'Data/out/validation/',
        x_col='images_png',
        y_col=['malignancy'],
        class_mode="raw",
        batch_size=BATCH_SIZE_VALIDATION,
        shuffle=False,
        seed=1,
        target_size=(target_w,target_h))

test_generator = data_generator.flow_from_dataframe(
        dataframe=df_test,
        directory='Data/out/test/',
        x_col='images_png',
        y_col=['malignancy'],
        class_mode="raw",
        batch_size=BATCH_SIZE_TESTING,
        shuffle=False,
        seed=1,
        target_size=(target_w,target_h))
'----------------------------END TEST-----------------------------------------'

'----------------------------STANDARDIZATION GENERATOR------------------------'

# # Image Data Generator
# train_generator = util.get_train_generator(df_train, IMAGE_DIR, 'images_png', 'malignancy',
#                                            shuffle=True, batch_size=8,
#                                            seed=1, target_w = 128, target_h = 128,
#                                            samplewise_center= False,
#                                            samplewise_std_normalization=False,
#                                            rescale= None )

# valid_generator, test_generator= util.get_test_and_valid_generator(df_val, df_test, df_train, 
#                                                                    'images_png', 'malignancy',
#                                                                    sample_size=100, batch_size=8,
#                                                                    seed=1, target_w = 128, target_h = 128,
#                                                                    samplewise_center=False,
#                                                                    samplewise_std_normalization=False,
#                                                                    rescale= None)
'-------------------------------END-------------------------------------------'

# Note: the plotted image is not what the NN sees since imshow treats negative values
# as black pixels.


# plot train image in batch
plt.figure(figsize=(40,40))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    # plot validation image in batch
    x_train, y_valid = train_generator.__getitem__(0)
    plt.imshow(x_train[i], cmap='gray')
    #plt.axis('off')

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()




'____________________________CHOSE MODEL______________________________________'
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

'------------------------------DENSENET121-------------------------------'
# # Sigmoid: probability independent between labels.
# # Softmax: probability must sum to 1 between all labels.


# ## DENSENET121 (pre-trained)

# # Softmax for binary classification.

# # path to weights
# densenet_weights_path = 'C:\\Users\\boels\\.keras\\models\\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'

# # create the base pre-trained model
# base_model = DenseNet121(weights=densenet_weights_path, 
#                          include_top=False)

# # not training first layer model as it is already trained
# for layer in base_model.layers:
#     layer.trainable = False

# # x is the output of the model
# x = base_model.output

# # a scalar value for each batch and channel to reduce the # of parameters before the FC.
# x = GlobalAveragePooling2D()(x)

# # and a logistic layer
# predictions = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=predictions)


# # Print the model summary
# model.summary()

# # opimizer
# OPTIMIZER = keras.optimizers.Adam(learning_rate=0.01)

# model.compile(loss= 'binary_crossentropy',
#               optimizer= OPTIMIZER,
#               metrics=['accuracy'])


'----------------------------------ResNet50--------------------------------------'

# from keras.applications.resnet import ResNet50

# ## DEFINE


# # path to weights without top layer
# resnet_weights_path = 'C:\\Users\\boels\\.keras\\models\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# # create the base pre-trained model
# base_model = ResNet50(weights=resnet_weights_path, 
#                       include_top=False)

# # not training first layer model as it is already trained
# for layer in base_model.layers:
#     layer.trainable = False

# x = base_model.output

# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)

# # and a logistic layer
# predictions = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=predictions)


# model.summary()


# ## COMPILE

# OPTIMIZER = keras.optimizers.Adam(learning_rate=0.01)

# # use custom weighted loss function or give class weigths in fit_generator
# model.compile(optimizer= OPTIMIZER,
#               loss= 'binary_crossentropy',
#               metrics=['accuracy'])
# # callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)

'----------------------------------VGG19--------------------------------------'
# from keras.applications import VGG19

# vgg19_weights_path = 'C:\\Users\\boels\\.keras\\models\\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

# base_model = VGG19(include_top=False, 
#                    weights=vgg19_weights_path)

# # not training first layer model as it is already trained
# for layer in base_model.layers:
#     layer.trainable = False

# x = base_model.output

# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)

# # and a logistic layer
# predictions = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.summary()


# ## COMPILE

# OPTIMIZER = keras.optimizers.Adam(learning_rate=0.01)

# # use custom weighted loss function or give class weigths in fit_generator
# model.compile(optimizer= OPTIMIZER,
#               loss= 'binary_crossentropy',
#               metrics=['accuracy'])

'----------------------------------ResNet50V2--------------------------------------'

from keras.applications.resnet_v2 import ResNet50V2

L_REG = 'l1'
LR = 0.0001 
OPTIMIZER = keras.optimizers.Adam(learning_rate=LR)

## DEFINE


# path to weights without top layer
resnet_weights_path = 'C:\\Users\\boels\\.keras\\models\\resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

# create the base pre-trained model
base_model = ResNet50V2(weights=resnet_weights_path, 
                      include_top=False)

# not training first layer model as it is already trained
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.1)(x)  # Regularize with dropout


# and a logistic layer
predictions = Dense(1, activation="sigmoid",
                    kernel_regularizer=L_REG)(x)

model = Model(inputs=base_model.input, outputs=predictions)

#model.summary()


## COMPILE

# use custom weighted loss function or give class weigths in fit_generator
model.compile(optimizer= OPTIMIZER,
              loss= 'binary_crossentropy',
              metrics=['accuracy'])


'--------------------------end model architecture--------------------'

## TRAINING

EPOCHS = 40
PATIENCE = 40


STEPS_PER_EPOCH_TRAINING = len(train_generator)
STEPS_PER_EPOCH_VALIDATION = len(valid_generator)
# callbacks
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=PATIENCE)

# the data is augmented accordingly to the number of batches as argument
history = model.fit(train_generator, 
                    validation_data= valid_generator,
                    steps_per_epoch=STEPS_PER_EPOCH_TRAINING, # batches
                    validation_steps=STEPS_PER_EPOCH_VALIDATION,# batches
                    epochs = EPOCHS,
                    callbacks= [early_stopping],
                    class_weight= class_weight)


## Visualization and Fine-Tuning

# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training and Validation Loss Curves with "+ base_model.name)
plt.show()

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train accuracy', 'Val accuracy'], loc='center right')
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title("Training Accuracy and Validation Accuracy with "+ base_model.name)
plt.show()




## LOADING WEIGHTS


model.load_weights('Data/weights/my_model_weights_Epochs40.h5')


## PREDICTION AND EVALUATION ON TEST SET


# predict
predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))
print(f' std: {predicted_vals.std()}\n min: {predicted_vals.min()}\n max: {predicted_vals.max()}')


#ROC and AUROC
labels = ['maligancy']
auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator, base_model)


## EXPORT VALUES

# selects columns 4, 1, and 3
df_test_eval = df_test[df_test.columns[[4, 1, 3]]]
# add predictions to df_test and save as .csv file  
df_test_eval['predictions'] = predicted_vals
df_test_eval.to_csv("Output/df_test.csv", index=None)



## GradCAM: Visualizing Learning 


layer_name = 'post_bn'

df = df_test
IMAGE_DIR = "Data/out/test/"

# only show the labels with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]

images = ['demd117_CC_Right.png', 'demd122_MLO_Right.png', 
          'demd1501_CC_Right.png', 'demd50449_MLO_Right.png']

for image in images:

    # choose image with positive label
    util.compute_gradcam(model, image, IMAGE_DIR, df, labels,
                         labels_to_show,
                         layer_name= layer_name)


'-----------------DEBUG GRADCAM--------------------------'







