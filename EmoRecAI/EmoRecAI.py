"""
---------------- EmoRecAI ----------------

Facial Emotion Recognition AI is a project that aims to develop an AI model capable
of recognizing and classifying human emotions based on facial expressions. The 
project involves collecting and preprocessing a dataset of facial images, training
a ML model on the dataset, and evaluating the model's performance and accuracy.

Based on Udemy course "Modern Artificial Intelligence Masterclass" by Dr. Ryan Ahmed

------------------------------------------
Revised by: Nick Felker

Creation Date: 2/11/2026
Last Date Modified: 2/12/2026

"""

# REQUIRED IMPORTS AND LIBRARIES
import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import random

# TensorFlow/Keras imports - cleaned and organized
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers, backend as K
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, 
    Activation, Add, Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
)

# IPython display (only needed in Jupyter notebooks)
try:
    from IPython.display import display
except ImportError:
    pass  # Not needed when running as script

"""

Load the dataset and get relevant info about the dataframe

"""
#Load facial key points dataset
# Use a raw string (or normalized path) so backslashes aren't treated as escape sequences
data_path = r'C:\Users\nick.felker\Downloads\Emotion AI Dataset\data.csv'
data_path = os.path.normpath(data_path)
facialKey_df = pd.read_csv(data_path)

#Obtain relavant info about the dataframe
facialKey_df.info()

#Check if null values exist in the dataframe and get the shape of the 'Image' column
facialKey_df.isnull().sum()
facialKey_df['Image'].shape

#Convert each image string into a 96x96 numpy array
facialKey_df['Image'] = facialKey_df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))

#Get the shape of the first image in the dataframe
facialKey_df['Image'].iloc[0].shape

#Examples to get relevant info about the dataframe
#print("Print outs for example data collection:")
#print(facialKey_df['right_eye_center_x'].describe())


"""

Visualize images from the dataframe to confirm expectations and functionality

"""
"""
#Visualize a random image with its corresponding key points and allow 0 as a valid index
i = np.random.randint(0, len(facialKey_df))
plt.figure(figsize=(3, 3))
plt.imshow(facialKey_df['Image'].iloc[i], cmap='gray')
for j in range(1, 31, 2):
    x = facialKey_df.iloc[i, j-1]
    y = facialKey_df.iloc[i, j]
    if np.isfinite(x) and np.isfinite(y):
        plt.plot(x, y, 'rx')
plt.axis('off')
plt.show()

#Visualize a grid of 16 images with their corresponding key points
fig = plt.figure(figsize=(8, 8))
for idx in range(64):
    ax = fig.add_subplot(8, 8, idx + 1)
    # use positional indexing for the image
    image = facialKey_df['Image'].iloc[idx]
    ax.imshow(image, cmap='gray')
    for j in range(1, 31, 2):
        x = facialKey_df.iloc[idx, j-1]
        y = facialKey_df.iloc[idx, j]
        if np.isfinite(x) and np.isfinite(y):
            ax.plot(x, y, 'rx')
    ax.axis('off')

plt.tight_layout()
plt.show()
"""

"""

Start to augment the images for further use

"""
#Create a copy of the dataframe to manipulate
facialKey_df_h = copy.copy(facialKey_df)
facialKey_df_v = copy.copy(facialKey_df)

#Get the columns of the dataframe excluding the 'Image' column
columns = facialKey_df_h.columns[:-1]

#Flip the image horizontally across the y axis
facialKey_df_h['Image'] = facialKey_df_h['Image'].apply(lambda x: np.flip(x, axis=1))
for i in range(len(columns)):
    if i%2 == 0:
        facialKey_df_h[columns[i]] = facialKey_df_h[columns[i]].apply(lambda x: 96. - float(x))

#Flip the image vertically across the x axis
facialKey_df_v['Image'] = facialKey_df_v['Image'].apply(lambda x: np.flip(x, axis=0))
for i in range(len(columns)):
    if i%2 == 1:
        facialKey_df_v[columns[i]] = facialKey_df_v[columns[i]].apply(lambda x: 96. - float(x))

"""
#Show the orignal image and the flipped image with their corresponding key points to confirm functionality
plt.imshow(facialKey_df['Image'].iloc[0], cmap='gray')
for j in range(1, 31, 2):
    x = facialKey_df.iloc[0, j-1]
    y = facialKey_df.iloc[0, j]
    if np.isfinite(x) and np.isfinite(y):
        plt.plot(x, y, 'rx')
plt.title('Original Image')
plt.axis('off')
plt.show()

plt.imshow(facialKey_df_h['Image'].iloc[0], cmap='gray')
for j in range(1, 31, 2):
    x = facialKey_df_h.iloc[0, j-1]
    y = facialKey_df_h.iloc[0, j]
    if np.isfinite(x) and np.isfinite(y):
        plt.plot(x, y, 'rx')
plt.title('Horizontally Flipped Image')
plt.axis('off')
plt.show()
"""

#Concat the original dataframe with the flipped dataframe to create an augmented dataset
facialKey_df_aug = pd.concat([facialKey_df, facialKey_df_h])
facialKey_df_aug = pd.concat([facialKey_df_aug, facialKey_df_v])
facialKey_df_aug.shape

#Randomly increase the brightness of the images in the orignal dataframe and concat the new
#dataframe with the already augmented dataframe to create an larger dataset
facialKey_df_copy = copy.copy(facialKey_df)
facialKey_df_copy['Image'] = facialKey_df_copy['Image'].apply(lambda x: np.clip(random.uniform(1.5, 2)*x, 0.0, 255.0))
facialKey_df_aug = pd.concat([facialKey_df_aug, facialKey_df_copy])
facialKey_df_aug.shape

"""
#Show the brightened images to confirm functionality
plt.imshow(facialKey_df_copy['Image'].iloc[0], cmap='gray')
for j in range(1, 31, 2):
    x = facialKey_df_copy.iloc[0, j-1]
    y = facialKey_df_copy.iloc[0, j]
    if np.isfinite(x) and np.isfinite(y):
        plt.plot(x, y, 'rx')
plt.title('Brightened Image')
plt.axis('off')
plt.show()
"""

"""

Data normalization and splitting into training and testing sets

"""
#Obtain the value of images from column 31 and normalize the images
img_series = facialKey_df_aug.iloc[:, 30]
# Convert Series of (96,96) arrays into a single numpy array of shape (n,96,96)
img = np.stack(img_series.values).astype('float32')
# Normalize
img = img / 255.0

#Create an array of shape (x, 96, 96, 1) to input into the model
X = np.expand_dims(img, axis=-1)  # shape (n, 96, 96, 1)
X = X.astype('float32')

#Obtain the x and y coordinates being used for the target (first 30 columns)
y = facialKey_df_aug.iloc[:, :30].values.astype('float32')

#Split the dataset into training and testing datasets and verify the split is correct
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)

"""

ResNet model architecture

"""
def res_block(X, filter, stage):

    #Convolutional block
    X_copy = X

    f1, f2, f3 = filter

    #Main path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res' + str(stage) + '_conv_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D((2, 2), padding='same')(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, (3, 3), strides=(1, 1), padding='same', name='res' + str(stage) + '_conv_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), name='res' + str(stage) + '_conv_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_c')(X)

    #Shortcut path
    X_copy = Conv2D(f3, (1, 1), strides=(1, 1), name='res' + str(stage) + '_conv_shortcut', kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPooling2D((2, 2), padding='same')(X_copy)
    X_copy = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_shortcut')(X_copy)

    #Add the main path and the shortcut path together
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    #Identity block 1
    X_copy = X

    #Main path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res' + str(stage) + '_id_1_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_id_1_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, (3, 3), strides=(1, 1), padding='same', name='res' + str(stage) + '_id_1_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_id_1_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), name='res' + str(stage) + '_id_1_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_id_1_c')(X)
    
    #Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    #Identity block 2
    X_copy = X

    #Main path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res' + str(stage) + '_id_2_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_id_2_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, (3, 3), strides=(1, 1), padding='same', name='res' + str(stage) + '_id_2_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_id_2_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), name='res' + str(stage) + '_id_2_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_id_2_c')(X)

    #Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X


#Input tensor shape
input_shape = (96, 96, 1)

#Input layer
X_input = Input(input_shape)

#Zero-Padding
X = layers.ZeroPadding2D((3, 3))(X_input)

#Stage 1
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPool2D((3, 3), strides=(2, 2))(X)

#Stage 2
X = res_block(X, [64, 64, 256], stage=2)

#Stage 3
X = res_block(X, [128, 128, 512], stage=3)

#Stage 4
X = res_block(X, [256, 256, 1024], stage=4)

#Average Pooling
X = GlobalAveragePooling2D(name='avg_pool')(X)

#Final layer
X = Flatten()(X)
X = Dense(4096, activation='relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation='relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation='relu')(X)

#Model instantiation
model_1_facialKeyPoints = Model(inputs=X_input, outputs=X)
model_1_facialKeyPoints.summary()