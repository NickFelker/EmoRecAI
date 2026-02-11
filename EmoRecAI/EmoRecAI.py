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
Last Date Modified: 2/11/2026

"""

#REQUIRED IMPORTS AND LIBRARIES
from numpy.random import f
import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import random


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


