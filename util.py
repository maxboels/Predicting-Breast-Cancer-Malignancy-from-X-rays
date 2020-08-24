# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile as tiff
import cv2
import os
from PIL import Image
import shutil, os
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import normalize
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import (roc_curve, accuracy_score, confusion_matrix)
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve


# Import data generator from keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model


def read_filenames(): 
    
    # training images
    local_path = ''
    folder_training_benign = 'Data/training/benign/'
    foler_training_cancerous = 'Data/training/cancerous/'
    
    files_benign = local_path+folder_training_benign
    files_cancerous = local_path+foler_training_cancerous
    
    
    train_list_files_benign = []
    train_list_files_cancerous  = []
    
    # create list benign train
    for file in os.listdir(files_benign):
        if file.endswith('.tif'):
            train_list_files_benign.append(file)
    
    # create list cancerous train
    for file in os.listdir(files_cancerous):
        if file.endswith('.tif'):
            train_list_files_cancerous.append(file)
            
    
    # testing images
    local_path = ''
    folder_testing_benign = 'Data/testing/benign/'
    foler_testing_cancerous = 'Data/testing/cancerous/'
    
    files_benign = local_path+folder_testing_benign
    files_cancerous = local_path+foler_testing_cancerous
    
    test_list_files_benign = []
    test_list_files_cancerous  = []
    
    for file in os.listdir(files_benign):
        if file.endswith('.tif'):
            test_list_files_benign.append(file)
            
    for file in os.listdir(files_cancerous):
        if file.endswith('.tif'):
            test_list_files_cancerous.append(file)  
      

    return train_list_files_benign, train_list_files_cancerous, test_list_files_benign, test_list_files_cancerous



def not_augmented_files(filenames):
    """
    Returns the selected images that ends with "Left.tif" or "Right.tif"
    
    Args:
        filenames (list): list of filenames.
        
    Returns: the raw filenames for each patient_id.
    
    """
    unique_files = [ids for ids in filenames if ids.endswith('Left.tif')\
                                               or ids.endswith('Right.tif')]
        
    return unique_files

def create_dataframes(filelists, train_b, train_c, test_b, test_c):
    
    """
    Returns dataframes with all the images names and their labels.
    
    Args:
        filelists (list): list of filenames.
        train_b
        train_c
        test_b
        test_c
        
    Returns: dataframes for each label and dataset.
    
    """
    
    df1 = pd.DataFrame({'images': train_b,
                    'patient_id': np.nan,
                    'array': np.nan,
                    'malignancy': [0] * len(train_b)})

    df2 = pd.DataFrame({'images': train_c,
                        'patient_id': np.nan,
                        'array': np.nan,
                        'malignancy': [1] * len(train_c)})
    
    df3 = pd.DataFrame({'images': test_b,
                        'patient_id': np.nan,
                        'array': np.nan,
                        'malignancy': [0] * len(test_b)})
    
    df4 = pd.DataFrame({'images': test_c,
                        'patient_id': np.nan,
                        'array': np.nan,
                        'malignancy': [1] * len(test_c)})
    
    df = [df1, df2, df3, df4]
    
    for count, file in enumerate(filelists):
        temp = [sub.split('d')[2] for sub in file] 
        df[count]['patient_id'] = [sub.split('_')[0] for sub in temp]
        

        
    return df1, df2, df3, df4


def create_datasets(df1, df2, df3, df4):
    """
    Args:
        df1 (dataframe): train benign
        df2 (dataframe): train cancerous
        df3 (dataframe): test benign
        df4 (dataframe): test cancerous
            
    Returns: test set with at least 50% of positives cases.
    """
    # merge df train labels
    df_train = pd.concat([df1, df2], ignore_index=True)
    
    # merge df test labels
    df_test = pd.concat([df3, df4], ignore_index=True)
    
    # merge all data and shuffle to balance the labels
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    duplicates = df[df.duplicated('images')]
    df = df.drop_duplicates('images')
    print(f'We found and dropped {len(duplicates)} duplicated values in the data set which are:')
    print(f'{duplicates}')
    
    # shuffle all the data
    df_shuffled = df.sample(frac=1)
    
    # create test set
    df_test_p = df_shuffled.loc[df_shuffled['malignancy'] == 1].sort_values(['patient_id'])[:65]
    df_test_n = df_shuffled.loc[df_shuffled['malignancy'] == 0].sort_values(['patient_id'])[:64]
    df_test = pd.concat([df_test_p, df_test_n])
    
    # create training set
    df_train_p = df_shuffled.loc[df_shuffled['malignancy'] == 1].sort_values(['patient_id'])[65:]
    df_train_n = df_shuffled.loc[df_shuffled['malignancy'] == 0].sort_values(['patient_id'])[64:]
    df_train = pd.concat([df_train_p, df_train_n])
    # same 'patient_id' for MLO and CC at split [65:66] and [64:65].
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    print('')
    print('Training and Testing DataFrame have been created.')
    print('')
    
    return df_train, df_test


def drop_overlapping_patients_train(df_train, df_test):

    patient_overlap = check_leakage(df_train, df_test, 'patient_id')
    
    # Find and Drop overlapping indexes
    overlap_index = []
    
    for idx in range(len(patient_overlap)):
        overlap_index.extend(df_train.index[df_train['patient_id'] == patient_overlap[idx]].tolist())
    
    # Drop the overlapping rows from the training set
    df_train.drop(overlap_index, inplace=True)
    
    # Reset indexes
    df_train.reset_index(drop=True, inplace=True)
    
    print(f'We dropped overlapping patients ids from the training set patient:{patient_overlap}')
    
    return df_train


def df_add_arrays(path, df):
    """
    Adds the .tiff files arrays.
    
    Parameters
    ----------
    path : string
        folder path
    df : data frame
    column : string
        file name column (images)
    Returns
    -------
    df : data frame
    """
    column = 'images'
    files = df[column].values.tolist()
    array_list = []
    
    for file in files:
        im = tiff.imread(path+file)
        array_list.append(im)
         
    df['array'] = array_list
    
    return df


def add_png_arrays(df):
    """
    Adds the normalized arrays into the new colmun 'norm_mammo' of the dataframe (df).
    
    Parameters
    ----------
    df : Data frame
        training or test set.

    Returns
    -------
    """
    files = df['images']
    array_list = []
    
    for file in files:
        # locate each array with the corresponding file name.
        mammogram = df.loc[df['images'] == file, 'array'].iloc[0]
        mammogram_uint8_by_cv2 = cv2.normalize(mammogram, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
        array_list.append(mammogram_uint8_by_cv2)
                
    # create new columns in df and add the lists.
    df['norm_mammo'] = np.nan
    df['norm_mammo'] = array_list

    return


def add_png_filenames(df):
    
    """
    Adds the normalized arrays into the new colmun 'norm_mammo' of the dataframe (df).
    
    Parameters
    ----------
    df : Data frame
        training or test set.

    Returns
    -------
    """
    files = df['images']
    file_names_list = []
    
    for file in files:        
        file = file[:-3]+'png'
        file_names_list.append(file)
        
    # create new columns in df and add the lists.
    df['images_png'] = ""
    df['images_png'] = file_names_list

    return



def show_normalized_mammo(df):
    
    idx = np.random.randint(len(df))
    mammogram_uint8_by_cv2 = df.norm_mammo[idx]
    
    plt.imshow(mammogram_uint8_by_cv2, interpolation='nearest')
    plt.show()

    return

def oversampling(df_train):

    print(df_train.malignancy.value_counts())
    
    max_size = df_train['malignancy'].value_counts().max()
    lst = [df_train]
    for class_index, group in df_train.groupby('malignancy'):
        lst.append(group.sample(max_size-len(group), replace=True))
    df_train_oversampled = pd.concat(lst)
    
    print(df_train_oversampled.malignancy.value_counts())
        
    return df_train_oversampled

def get_class_weights(df):
    
    neg, pos = np.bincount(df['malignancy'])
    total = neg + pos
    
    # get weights
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    # get dict
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    return class_weight


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        
        # for each class, add average weighted loss for that class 
        loss += -1 * (K.mean(pos_weights * y_true * K.log(y_pred + epsilon) \
                          + neg_weights * (1 - y_true) * K.log(1-y_pred + epsilon)))
        
        return loss
    
    return weighted_loss


def check_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    patients_in_both_groups = list(df1_patients_unique.intersection(df2_patients_unique))
    print(f'These patients are in both the training and test datasets: {patients_in_both_groups}')
    n_overlap = len(patients_in_both_groups)
    print(f'There are {n_overlap} Patient IDs in both the training and test sets')
    print('')
    leakage = n_overlap > 0

    return patients_in_both_groups

def plot_pixel_values(df):

    idx = np.random.randint(len(df))
    im = df.array[idx]
    sns.distplot(im.ravel(), 
                 label=f'Pixel Mean {np.mean(im):.2f}', 
                 kde=False)
    
    plt.legend(loc='best')
    plt.title('Distribution of Pixel Intensities in the Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# Pixels in Image')
    
    print('Histogram of pixel distribution from a random image')

def plot_label_frequency(df1, df2):
    
    """
    Returns the label frequency for each dataset

    Args:
        df1 (dataframe): dataframe describing first dataset (train)
        df2 (dataframe): dataframe describing second dataset(test)
    
    Return: plot.
        
    """
    
    train_freq_pos = [df1.loc[df1['malignancy']==1].shape[0]]
    train_freq_neg = df1.loc[df1['malignancy']==0].shape[0]
    test_freq_pos = df2.loc[df2['malignancy']==1].shape[0]
    test_freq_neg = df2.loc[df2['malignancy']==0].shape[0]
    
    data = pd.DataFrame({"Dataset": "train", "Label": "Positive", "Images": train_freq_pos})
    data = data.append({"Dataset": "train", "Label": "Negative", "Images": train_freq_neg}, ignore_index=True)
    data = data.append({"Dataset": "test", "Label": "Positive", "Images": test_freq_pos}, ignore_index=True)
    data = data.append({"Dataset": "test", "Label": "Negative", "Images": test_freq_neg}, ignore_index=True)
    
    f = sns.barplot(x="Dataset", y="Images", hue="Label" ,data=data)
    f.set_title('Label Frequency')
    
    print(f'There are {train_freq_pos} positive and {train_freq_neg} negative training labels.')
    
    return


def show_tif(local_folder, df):
    
    idx = np.random.randint(len(df))
    file_name = df.images[idx]
    im = tiff.imread(local_folder+file_name)
    cv2.imshow('Title', im.astype(np.float32)/np.max(im))
    cv2.waitKey(0)
    
    return


def show_cv2(local_folder, file_name):
    
    im = cv2.imread(local_folder+file_name, -1)
    cv2.imshow('Title', im.astype(np.float32)/np.max(im))
    cv2.waitKey(0)
    
    return


def move_raw_files(source, dest, df):
    """
    Description: Move all raw files to the train or test folder.
    source: 
    
    info: df is df.column (images)
    """
    # use .values.tolist() method to convert series to list.
    files = df.values.tolist()
    
    for f in files:
        shutil.move(source+f, dest)
    
    return

def save_to_png(df, folder):
    """
    Parameters
    ----------
    df : data frame
        train or test data frame.
    folder : string
        train or test destination folder.
    """
    files = df['images']
    
    for count, file in enumerate(files):  
        array = df.norm_mammo[count]
        img = Image.fromarray(array,'RGB')
        img.save('Data/out/'+folder+'/'+file[:-3]+'png')

    return


def create_df_val(df_train):


    # filter by positives and negatives labels and sort by patient (avoid overlap)
    df_val_p = df_train.loc[df_train['malignancy'] == 1].sort_values(['patient_id'])[:25]
    df_val_n = df_train.loc[df_train['malignancy'] == 0].sort_values(['patient_id'])[:25]
    df_val = pd.concat([df_val_p, df_val_n])    
    
    df_train_p = df_train.loc[df_train['malignancy'] == 1].sort_values(['patient_id'])[25:]
    df_train_n = df_train.loc[df_train['malignancy'] == 0].sort_values(['patient_id'])[25:]
    df_train = pd.concat([df_train_p, df_train_n])
    
    # shuffle
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    return df_val, df_train


def show_png_images(df, label, IM_DIR):
    """
    Parameters
    ----------
    df : dataframe
    label : int
        0 or 1
    Returns
    -------
    images
    """

    df_class = df.loc[df.malignancy == label]
    
    # Extract numpy values from Image column in data frame
    images = df_class['images_png'].values
    
    # Extract 16 random images from it
    random_images = np.random.choice(images, size=25, replace=False).tolist()    
    # Location of the image dir
    img_dir = IM_DIR
    
    print(f'Display Random Images from class: {label}')
    
    # Adjust the size of your images
    plt.figure(figsize=(40,40))
    
    # Iterate and plot random images
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        img = plt.imread(os.path.join(img_dir, random_images[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()
    
    return 


def find_duplicates(my_list):
    import collections
    print([item for item, count in collections.Counter(my_list).items() if count > 1])
    return



def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8,
                        seed=1, target_w = 128, target_h = 128,
                        samplewise_center=True, samplewise_std_normalization=None,
                        rescale= None):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images either between -1 and 1 or between 0 and 1.
    image_generator = ImageDataGenerator(
        samplewise_center= samplewise_center, #Set each sample mean to 0.
        samplewise_std_normalization= samplewise_std_normalization, # Divide each input by its standard deviation
        rotation_range = 360,
        horizontal_flip= True,
        vertical_flip= True,
        rescale= rescale # Normalize between [0:1].
        )
    
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=[y_cols],
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


def get_test_and_valid_generator(valid_df, test_df, train_df, x_col, \
                                 y_cols, sample_size=100, batch_size=8, seed=1, \
                                target_w = 320, target_h = 320,
                                samplewise_center=True, samplewise_std_normalization=True,
                                rescale= None):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe= train_df, 
        directory='Data/out/train/', 
        x_col='images_png', 
        y_col= ['malignancy'], 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0] # length = sample_size

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        samplewise_center= samplewise_center, #Set each sample mean to 0.
        samplewise_std_normalization= samplewise_std_normalization, # Divide each input by its standard deviation
        rotation_range = 0,
        horizontal_flip= False,
        vertical_flip= False,
        rescale= rescale # Normalize between [0:1].
        )
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory= 'Data/out/validation/',
            x_col=x_col,
            y_col=[y_cols],
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory='Data/out/test/',
            x_col=x_col,
            y_col=[y_cols],
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    
    return valid_generator, test_generator


def compute_label_freq(df):
    """
    Parameters
    ----------
    df : dataframe or object
    labels : list of string
    
    Returns
    -------
    freq_pos : scalar
    freq_neg : scalar

    """    
    # labels in training set
    freq_pos = np.sum(df['malignancy'] == 1, axis = 0)
    freq_neg = np.sum(df['malignancy'] == 0, axis = 0)
    
    return freq_pos, freq_neg




def get_roc_curve(labels, predicted_vals, generator, base_model):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " AUC: " + str(round(auc_roc, 3)))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve with '+ base_model.name)
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals



def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)
    label = df.loc[df['images_png'] == img, 'malignancy'].iloc[0]

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title(f"Original class: {label}")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["images_png"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std


def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


'--------------------Week_2--------METRICS_and_EVALUATION----------------------'


def get_true_pos(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def get_true_neg(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def get_false_neg(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def get_false_pos(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))



def get_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)

    columns = ["", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        df.loc[i][1] = round(tp(y[:, i], pred[:, i], thresholds[i]),
                             3) if tp != None else "Not Defined"
        df.loc[i][2] = round(tn(y[:, i], pred[:, i], thresholds[i]),
                             3) if tn != None else "Not Defined"
        df.loc[i][3] = round(fp(y[:, i], pred[:, i], thresholds[i]),
                             3) if fp != None else "Not Defined"
        df.loc[i][4] = round(fn(y[:, i], pred[:, i], thresholds[i]),
                             3) if fn != None else "Not Defined"
        df.loc[i][5] = round(acc(y[:, i], pred[:, i], thresholds[i]),
                             3) if acc != None else "Not Defined"
        df.loc[i][6] = round(prevalence(y[:, i]),
                             3) if prevalence != None else "Not Defined"
        df.loc[i][7] = round(sens(y[:, i], pred[:, i], thresholds[i]),
                             3) if sens != None else "Not Defined"
        df.loc[i][8] = round(spec(y[:, i], pred[:, i], thresholds[i]),
                             3) if spec != None else "Not Defined"
        df.loc[i][9] = round(ppv(y[:, i], pred[:, i], thresholds[i]),
                             3) if ppv != None else "Not Defined"
        df.loc[i][10] = round(npv(y[:, i], pred[:, i], thresholds[i]),
                              3) if npv != None else "Not Defined"
        df.loc[i][11] = round(auc(y[:, i], pred[:, i]),
                              3) if auc != None else "Not Defined"
        df.loc[i][12] = round(f1(y[:, i], pred[:, i] > thresholds[i]),
                              3) if f1 != None else "Not Defined"
        df.loc[i][13] = round(thresholds[i], 3)

    df = df.set_index("")
    
    return df

def get_accuracy(y, pred, th=0.5):
    """
    Compute accuracy of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        accuracy (float): accuracy of predictions at threshold
    """
    accuracy = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get TP, FP, TN, FN using our previously defined functions
    TP = get_true_pos(y, pred, th)
    FP = get_false_pos(y, pred, th)
    TN = get_true_neg(y, pred, th)
    FN = get_false_neg(y, pred, th)

    # Compute accuracy using TP, FP, TN, FN
    accuracy = (TP + TN) / (TP + FP + TN + FN)
        
    return accuracy


def get_prevalence(y):
    """
    Compute prevalence.

    Args:
        y (np.array): ground truth, size (n_examples)
    Returns:
        prevalence (float): prevalence of positive cases
    """
    prevalence = 0.0
        
    prevalence = sum(y) / len(y)
        
    return prevalence

def get_sensitivity(y, pred, th=0.5):
    """
    Compute sensitivity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        sensitivity (float): probability that our test outputs positive given that the case is actually positive
    """
    sensitivity = 0.0
        
    # get TP and FN using our previously defined functions
    TP = get_true_pos(y, pred, th)
    FN = get_false_neg(y, pred, th)
    
    # use TP and FN to compute sensitivity
    sensitivity = TP / (TP + FN)
        
    return sensitivity

def get_specificity(y, pred, th=0.5):
    """
    Compute specificity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        specificity (float): probability that the test outputs negative given that the case is actually negative
    """
    specificity = 0.0
        
    # get TN and FP using our previously defined functions
    FP = get_false_pos(y, pred, th)
    TN = get_true_neg(y, pred, th)
    
    # use TN and FP to compute specificity 
    specificity = TN / (TN + FP)
        
    return specificity



def get_ppv(y, pred, th=0.5):
    """
    Compute PPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        PPV (float): positive predictive value of predictions at threshold
    """
    PPV = 0.0
        
    # get TP and FP using our previously defined functions
    TP = get_true_pos(y, pred, th)
    FP = get_false_pos(y, pred, th)

    # use TP and FP to compute PPV
    PPV = TP / (TP + FP)
        
    return PPV



def get_npv(y, pred, th=0.5):
    """
    Compute NPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        NPV (float): negative predictive value of predictions at threshold
    """
    NPV = 0.0
    
    # get TN and FN using our previously defined functions
    TN = get_true_neg(y, pred, th)
    FN = get_false_neg(y, pred, th)

    # use TN and FN to compute NPV
    NPV = TN / (TN + FN)
    
    
    return NPV


def get_curve(gt, pred, target_names, curve='roc'):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            title = "ROC Curve and AUC"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg. Prec.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("Precision-Recall Curve")
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)





def plot_calibration_curve(y, pred, class_labels):
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()

'-------------------------NOT USED------------Interim Report-------------------'

def read_files(): 
    
    # training images
    local_path = ''
    folder_training_benign = 'Data/training/benign/'
    foler_training_cancerous = 'Data/training/cancerous/'
    
    files_benign = local_path+folder_training_benign
    files_cancerous = local_path+foler_training_cancerous
    
    train_list_arrays_ben = []
    train_list_arrays_can = []
    
    train_list_files_benign = []
    train_list_files_cancerous  = []
    
    for file in os.listdir(files_benign):
        if file.endswith('.tif'):
            train_list_files_benign.append(file)
            
    for file in os.listdir(files_cancerous):
        if file.endswith('.tif'):
            train_list_files_cancerous.append(file)  
    
    for file in train_list_files_benign:
        im = tiff.imread(local_path+folder_training_benign+file)
        train_list_arrays_ben.append(im)
    for file in train_list_files_cancerous:
        im = tiff.imread(local_path+foler_training_cancerous+file)
        train_list_arrays_can.append(im) 
    
    
    # testing images
    local_path = ''
    folder_testing_benign = 'Data/testing/benign/'
    foler_testing_cancerous = 'Data/testing/cancerous/'
    
    files_benign = local_path+folder_testing_benign
    files_cancerous = local_path+foler_testing_cancerous
    
    test_list_arrays_ben = []
    test_list_arrays_can = []
    
    list_files_benign = []
    list_files_cancerous  = []
    
    for file in os.listdir(files_benign):
        if file.endswith('.tif'):
            list_files_benign.append(file)
            
    for file in os.listdir(files_cancerous):
        if file.endswith('.tif'):
            list_files_cancerous.append(file)  
    
    for file in list_files_benign:
        im = tiff.imread(local_path+folder_testing_benign+file)
        test_list_arrays_ben.append(im)
    for file in list_files_cancerous:
        im = tiff.imread(local_path+foler_testing_cancerous+file)
        test_list_arrays_can.append(im)  

    
    return train_list_arrays_ben, train_list_arrays_can, test_list_arrays_ben, test_list_arrays_can


def create_datasets_old_version():
    train_list_arrays_ben, train_list_arrays_can, test_list_arrays_ben, test_list_arrays_can = read_files()
    # train
    train_ben = np.array(train_list_arrays_ben) #/ 65535
    train_can = np.array(train_list_arrays_can) #/65535
    train_all = np.concatenate((train_ben, train_can), axis = 0)
    
    # test
    test_ben = np.array(test_list_arrays_ben) #/ 65535
    test_can = np.array(test_list_arrays_can) #/ 65535
    test_all = np.concatenate((test_ben, test_can), axis = 0)
    
    #Note: lower accuracy with Normalized images.
    
    
    return train_ben, train_can, train_all, test_ben, test_can, test_all



def dataset_onehot():

    train_list_arrays_ben, train_list_arrays_can, test_list_arrays_ben, test_list_arrays_can = read_files()
    train_ben, train_can, train_all, test_ben, test_can, test_all = create_datasets()
    
    
    # training
    # create one hot labels for 2 classes
    labels_ben = np.concatenate((np.zeros((len(train_list_arrays_ben),1), dtype=int),np.ones((len(train_list_arrays_ben), 1), dtype=int)), axis=1)
    labels_can = np.concatenate((np.ones((len(train_list_arrays_can),1), dtype=int),np.zeros((len(train_list_arrays_can), 1), dtype=int)), axis=1)
    train_all_labels = np.concatenate((labels_ben, labels_can), axis=0 )
    
    # shuffle images and their label together
    inds = np.random.permutation(train_all.shape[0])
    train_X_onehot = train_all[inds, ...]
    train_y_onehot = train_all_labels[inds, :]
    
    
    # testing
    # create one hot labels for 2 classes
    test_ben = np.concatenate((np.zeros((len(test_list_arrays_ben),1), dtype=int),np.ones((len(test_list_arrays_ben), 1), dtype=int)), axis=1)
    test_can = np.concatenate((np.ones((len(test_list_arrays_can),1), dtype=int),np.zeros((len(test_list_arrays_can), 1), dtype=int)), axis=1)
    test_all_labels = np.concatenate((test_ben, test_can), axis=0 )
    
    # shuffle images and their label together
    inds = np.random.permutation(test_all.shape[0])
    test_X_onehot = test_all[inds, ...]
    test_y_onehot = test_all_labels[inds, :]    
    

    return train_X_onehot, train_y_onehot, test_X_onehot, test_y_onehot





def dataset_binary():
    
    train_ben, train_can, train_all, test_ben, test_can, test_all = create_datasets()
    
    # training
    labels_ben = np.zeros((train_ben.shape[0],1), dtype=int)
    labels_can = np.ones((train_can.shape[0],1),dtype=int)
    train_all_labels = np.concatenate((labels_ben, labels_can), axis=0 )
    
    # shuffle images and their label together
    inds = np.random.permutation(train_all.shape[0])
    train_X = train_all[inds, ...]
    train_y = train_all_labels[inds, :]
    
    # test
    labels_ben = np.zeros((test_ben.shape[0],1), dtype=int)
    labels_can = np.ones((test_can.shape[0],1),dtype=int)
    test_all_labels = np.concatenate((labels_ben, labels_can), axis=0 )
    
    # shuffle images and their label together
    inds = np.random.permutation(test_all.shape[0])
    test_X = test_all[inds, ...]
    test_y = test_all_labels[inds, :]  
    
    return train_X, train_y, test_X, test_y
    
    
def plot_training(fit_history):

    plt.figure(1, figsize = (15,8)) 
        
    plt.subplot(2, 2, 1)  
    plt.plot(fit_history.history['accuracy'])  
    plt.plot(fit_history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
        
    plt.subplot(2, 2, 2)  
    plt.plot(fit_history.history['loss'])  
    plt.plot(fit_history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
    
    plt.show()
    
    
def class_predicted(y_prediction):
    class_predicted = []
    for patient in range(len(y_prediction)):
        class_y= int
        if y_prediction[patient] >= 0.5:
            class_y = 1
        else:
            class_y = 0
        class_predicted.append(class_y)
    
    return class_predicted
    
def plot_roc_curve(test_y, y_prediction):
    
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(test_y, y_prediction)
    roc_auc = metrics.accuracy_score(test_y, np.round(y_prediction))
    
    plt.figure()
    lw = 2
    plt.plot(fpr_roc, tpr_roc, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def confusion_matrix_test(test_y, y_prediction):
    
    cm  = confusion_matrix(test_y, np.round(y_prediction))
    plt.figure()
    plot_confusion_matrix(cm, show_absolute=True,
                        show_normed=True, colorbar=True,
                        figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
    
    plt.xticks(range(2), ['Benign', 'Malignant'], fontsize=16)
    plt.yticks(range(2), ['Benign', 'Malignant'], fontsize=16)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return cm
    
   
    
    
def precision_recall(test_y, y_prediction):
    # 0.5 is rounded to zero.
    cm = confusion_matrix(test_y, np.round(y_prediction))
    
    true_negative, false_positive, false_negative, true_positive  = cm.ravel()
    
    # Sensitivity
    precision = true_positive / (true_positive + false_positive)
    
    # Specificity
    recall = true_positive / (true_positive + false_negative)
    
    print('Precison of breast CT scan for Malignancy:{:.2f}'.format(precision))
    print('Recall of breast CT scan for Malignancy:{:.2f}'.format(recall))
    
    
    return precision, recall


