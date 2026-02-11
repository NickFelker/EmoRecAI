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


"""

Load the dataset and get relevant info about the dataframe

"""
#Load facial key points dataset
# Use a raw string (or normalized path) so backslashes aren't treated as escape sequences
data_path = r'C:\Users\nick.felker\source\repos\Emotion AI Dataset\data.csv'
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
facialKey_df_copy = copy.copy(facialKey_df)

#Get the columns of the dataframe excluding the 'Image' column
columns = facialKey_df_copy.columns[:-1]

#Flip the image horizontally across the y axis
facialKey_df_copy['Image'] = facialKey_df_copy['Image'].apply(lambda x: np.flip(x, axis=1))
for i in range(len(columns)):
    if i%2 == 0:
        facialKey_df_copy[columns[i]] = facialKey_df_copy[columns[i]].apply(lambda x: 96. - float(x))
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

plt.imshow(facialKey_df_copy['Image'].iloc[0], cmap='gray')
for j in range(1, 31, 2):
    x = facialKey_df_copy.iloc[0, j-1]
    y = facialKey_df_copy.iloc[0, j]
    if np.isfinite(x) and np.isfinite(y):
        plt.plot(x, y, 'rx')
plt.title('Flipped Image')
plt.axis('off')
plt.show()
"""


