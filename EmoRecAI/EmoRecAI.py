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
Last Date Modified: 2/17/2026

"""

# REQUIRED IMPORTS AND LIBRARIES
from pyexpat import model
from tabnanny import verbose
from tkinter import LabelFrame
from turtle import shearfactor
import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import copy
import random
import json
import shutil
import subprocess
import time
import requests

# TensorFlow/Keras/Sklearn imports - cleaned and organized
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
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
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# IPython display (only needed in Jupyter notebooks)
try:
    from IPython.display import display
except ImportError:
    pass  # Not needed when running as script

# Define paths for model outputs (outside project directory)
MODEL_OUTPUT_DIR = r'C:\Users\nick.felker\Downloads\EmoRecAI_Models'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
weights_path = os.path.join(MODEL_OUTPUT_DIR, 'FacialKeyPoints_weights.keras')
emo_weights_path = os.path.join(MODEL_OUTPUT_DIR, 'FacialExpression_weights.keras')
model_json_path = os.path.join(MODEL_OUTPUT_DIR, 'FacialKeyPoints-model.json')
emo_model_json_path = os.path.join(MODEL_OUTPUT_DIR, 'FacialExpression-model.json')

#Flags to visualize data and train the models - set to 0 to skip visualization or training
visualize_data = 0
train_keypoint_model = 1
train_emo_model = 1


"""

Load the dataset and get relevant info about the dataframe

"""
#Load facial key points dataset
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
if visualize_data == 1:
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
        #Use positional indexing for the image
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

if visualize_data == 1:
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

    plt.imshow(facialKey_df_v['Image'].iloc[0], cmap='gray')
    for j in range(1, 31, 2):
        x = facialKey_df_v.iloc[0, j-1]
        y = facialKey_df_v.iloc[0, j]
        if np.isfinite(x) and np.isfinite(y):
            plt.plot(x, y, 'rx')
    plt.title('Vertically Flipped Image')
    plt.axis('off')
    plt.show()


#Concat the original dataframe with the flipped dataframe to create an augmented dataset
facialKey_df_aug = pd.concat([facialKey_df, facialKey_df_h])
facialKey_df_aug = pd.concat([facialKey_df_aug, facialKey_df_v])
facialKey_df_aug.shape

#Randomly increase the brightness of the images in the orignal dataframe and concat the new
#dataframe with the already augmented dataframe to create an larger dataset
facialKey_df_copy = copy.copy(facialKey_df)
facialKey_df_copy['Image'] = facialKey_df_copy['Image'].apply(lambda x: np.clip(random.uniform(1.5, 2.0)*x, 0.0, 255.0))
facialKey_df_aug = pd.concat([facialKey_df_aug, facialKey_df_copy])
facialKey_df_aug.shape

if visualize_data == 1:
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

Data normalization and splitting into training and testing sets

"""
#Obtain the value of images from column 31 and normalize the images
img_series = facialKey_df_aug.iloc[:, 30]
img = np.stack(img_series.values).astype('float32')
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
    X = MaxPool2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, (3, 3), strides=(1, 1), padding='same', name='res' + str(stage) + '_conv_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), name='res' + str(stage) + '_conv_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn' + str(stage) + '_conv_c')(X)

    #Shortcut path
    X_copy = Conv2D(f3, (1, 1), strides=(1, 1), name='res' + str(stage) + '_conv_shortcut', kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPooling2D((2, 2))(X_copy)
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


"""

Model architecture and training

"""
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
#X = res_block(X, [32, 32, 128], stage=2)

#Stage 3
X = res_block(X, [64, 64, 256], stage=2)

#Stage 4
X = res_block(X, [128, 128, 512], stage=3)

#Stage 5
#X = res_block(X, [256, 256, 1024], stage=5)

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

if train_keypoint_model == 1:
    #ADAM optimizer
    print("Compiling the model...")
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model_1_facialKeyPoints.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    #Save the best model with least validation loss
    checkpoint = ModelCheckpoint(filepath=weights_path, monitor="val_accuracy", verbose=1, save_best_only=True)
    history = model_1_facialKeyPoints.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1, callbacks=[checkpoint])

    #Save the model architecture and weights to json
    model_json = model_1_facialKeyPoints.to_json()
    with open(model_json_path, 'w') as json_file:
        json_file.write(model_json)

    print(f"Model training complete and saved to: {MODEL_OUTPUT_DIR}")

    if visualize_data == 1:
        #Get the model keys
        history.history.keys()

        #Plot the training and validation loss over epochs
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.show()

#Load the best model
with open(model_json_path, 'r') as json_file:
    json_savedModel = json_file.read()

model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
model_1_facialKeyPoints.load_weights(weights_path)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialKeyPoints.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

#Evaluate the model on the test set
result = model_1_facialKeyPoints.evaluate(X_test, y_test)
print(f"Test Loss: {result[0]}, Test Accuracy: {result[1]}")


"""

Facial expression recognition dataset loading and visualization

"""
#Read the csv for facial expression data
facialExp_df = pd.read_csv(r'C:\Users\nick.felker\Downloads\Emotion AI Dataset\icml_face_data.csv')
facialExp_df.head()
facialExp_df[' pixels'][0]

#Function to convert pixel values in string format to array format
def stoa(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

#Function to resize the images to 96x96
def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)

#Convert the pixel values to array format and resize the images
facialExp_df[' pixels'] = facialExp_df[' pixels'].apply(lambda x: stoa(x))
facialExp_df[' pixels'] = facialExp_df[' pixels'].apply(lambda x: resize(x))
facialExp_df.head()
facialExp_df.info()

#Check the shape of the dataframe and null values
facialExp_df.shape
facialExp_df.isnull().sum()

#Labels for the facial expression dataset
label_to_txt = {0: 'Angry', 1: 'Disgust', 2: 'Sad', 3: 'Happiness', 4: 'Surprise'}
emo = [0, 1, 2, 3, 4]

#Visualize the first image of each emotion to verify functionality
if visualize_data == 1:
    for i in emo:
        data = facialExp_df[facialExp_df['emotion'] == i][:1]
        img = data[' pixels'].item()
        img = img.reshape(96, 96)
        plt.imshow(img, cmap='gray')
        plt.title(label_to_txt[i])
        plt.axis('off')
        plt.show()

    facialExp_df.emotion.value_counts().index
    facialExp_df.emotion.value_counts()
    plt.figure(figsize=(10, 10))
    sns.barplot(x = facialExp_df.emotion.value_counts().index, y = facialExp_df.emotion.value_counts())
    plt.show()


"""

Data prep and image augmentation for facial expression dataset

"""
#Split the dataframe into features and lables
X = facialExp_df[' pixels']
y = to_categorical(facialExp_df['emotion'])

X = np.stack(X, axis=0)
X = X.reshape(24568, 96, 96, 1)

print(X.shape, y.shape)

#Split the dataset into training and testing datasets and verify the split is correct
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
print(X_val.shape, y_val.shape)

#Image preprocessing and augmentation
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

train_datagen = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [1.1, 1.5],
    fill_mode = "nearest")


"""

Build and train deep learning model for facial expression recognition using ResNet50 Transfer Learning

"""
# Get class distribution and print it for debugging
y_train_labels = np.argmax(y_train, axis=1)
unique, counts = np.unique(y_train_labels, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

# Don't use class weights for transfer learning - let the model learn naturally
print("Using ResNet50 Transfer Learning for superior performance")

# Convert grayscale to RGB for pretrained models
print("Converting grayscale images to RGB...")
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_val_rgb = np.repeat(X_val, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

# Resize to 224x224 for optimal ResNet50 performance
def resize_for_resnet(X):
    print(f"Resizing {X.shape[0]} images to 224x224...")
    X_resized = np.zeros((X.shape[0], 224, 224, 3), dtype=np.float32)
    for i in range(X.shape[0]):
        # Resize each channel
        X_resized[i] = cv2.resize(X[i], (224, 224))
    return X_resized

print("Resizing images for ResNet50...")
X_train_resized = resize_for_resnet(X_train_rgb)
X_val_resized = resize_for_resnet(X_val_rgb)
X_test_resized = resize_for_resnet(X_test_rgb)

input_shape = (224, 224, 3)

# Use pretrained ResNet50 as base model
print("Loading pretrained ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model initially
base_model.trainable = False

# Build the emotion classification model
X_input = Input(input_shape)
X = base_model(X_input, training=False)
X = GlobalAveragePooling2D()(X)
X = Dense(512, activation='relu')(X)
X = Dropout(0.2)(X)
X = Dense(256, activation='relu')(X)
X = Dropout(0.1)(X)
X = Dense(5, activation='softmax', name='emotion_output')(X)

#Model instantiation
model_2_emo = Model(inputs=X_input, outputs=X, name='ResNet50_Emotion')
model_2_emo.summary()

#Train the model
if train_emo_model == 1:
    print("Starting ResNet50 Transfer Learning Training...")
    
    # Phase 1: Train only the classifier head
    print("Phase 1: Training classifier head with frozen ResNet50...")
    model_2_emo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    #Early stopping with patience
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=True)

    #Learning rate reduction
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

    #Save the best model with highest validation accuracy
    checkpoint = ModelCheckpoint(filepath=emo_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Updated data generator for better augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=5,      # Reduce from 10
        width_shift_range=0.02, # Reduce from 0.05
        height_shift_range=0.02,
        zoom_range=0.02,
        horizontal_flip=True,
        brightness_range=[0.95, 1.05],  # More conservative
        # Remove shear - faces shouldn't be sheared
    )

    #History callback without class weights for better overall performance
    history_emo_phase1 = model_2_emo.fit(
        train_datagen.flow(X_train_resized, y_train, batch_size=32), 
        validation_data=(X_val_resized, y_val), 
        epochs=15, 
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # Phase 2: Fine-tune the last few layers of ResNet50
    print("Phase 2: Fine-tuning ResNet50 layers...")
    base_model.trainable = True
    
    # Freeze early layers, only train the last few layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Lower learning rate for fine-tuning
    model_2_emo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    history_emo_phase2 = model_2_emo.fit(
        train_datagen.flow(X_train_resized, y_train, batch_size=32), 
        validation_data=(X_val_resized, y_val), 
        epochs=10, 
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    #Save the model
    emo_model_json = model_2_emo.to_json()
    with open(emo_model_json_path, 'w') as json_file:
        json_file.write(emo_model_json)

    print(f"ResNet50 Transfer Learning model training complete and saved to: {MODEL_OUTPUT_DIR}")

#Load the best model
with open(emo_model_json_path, 'r') as json_file:
    emo_json_savedModel = json_file.read()

model_2_emo = tf.keras.models.model_from_json(emo_json_savedModel)
model_2_emo.load_weights(emo_weights_path)
model_2_emo.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

#Evaluate the model on the test set with resized images
score = model_2_emo.evaluate(X_test_resized, y_test)
print('ResNet50 Transfer Learning Test Accuracy: {:.2%}'.format(score[1]))


# Update prediction arrays to use resized images for evaluation
predicted_classes = np.argmax(model_2_emo.predict(X_test_resized), axis=-1)   
y_true = np.argmax(y_test, axis=-1)

cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()

L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize=(24, 24))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].squeeze(), cmap='gray')
    axes[i].set_title(f"True: {label_to_txt[y_true[i]]}\nPredicted: {label_to_txt[predicted_classes[i]]}")
    axes[i].axis('off')

plt.subplots_adjust(wspace=1)
plt.show()

print(classification_report(y_true, predicted_classes))


"""

Prediction function to incorporate both models for facial key point detection and facial expression recognition

"""
def predict(X_test):
    #Predict facial keypoints (still uses original 96x96 images)
    df_predict = model_1_facialKeyPoints.predict(X_test)

    # Convert and resize images for emotion prediction with ResNet50
    X_test_rgb = np.repeat(X_test, 3, axis=-1)
    X_test_resized = np.zeros((X_test.shape[0], 224, 224, 3), dtype=np.float32)
    for i in range(X_test.shape[0]):
        X_test_resized[i] = cv2.resize(X_test_rgb[i], (224, 224))
    
    #Predict facial expression with ResNet50
    df_emo_predict = np.argmax(model_2_emo.predict(X_test_resized), axis=-1)   

    #Reshaping array from (856,) to (856, 1) for concatenation
    df_emo_predict = np.expand_dims(df_emo_predict, axis=-1)

    #Converting the predicted key points and emotions into a dataframe
    df_predict = pd.DataFrame(df_predict, columns=columns)

    #Adding emotion predictions to the dataframe
    df_predict['emotion'] = df_emo_predict

    return df_predict

#Example usage of the prediction function
df_predict = predict(X_test)
df_predict.head()

# Plotting the test images and their predicted keypoints and emotions
fig, axes = plt.subplots(4, 4, figsize = (24, 24))
axes = axes.ravel()

for i in range(16):

    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title(f"ResNet50 Predicted Emotion: {label_to_txt[df_predict.loc[i]['emotion']]}")
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')


"""

Deploy the model - Currently unused and inoperable due to environment limitations

"""
"""
def deploy(directory, model):
    MODEL_DIR = directory
    version = 1

    export_path = os.path.join(MODEL_DIR, str(version))
    print(f"export_path = {export_path}")

    if os.path.isdir(export_path):
        print('Already saved a model, cleaning up the folder')
        shutil.rmtree(export_path)

    tf.saved_model.save(model, export_path)

    os.environ['MODEL_DIR'] = MODEL_DIR
    return MODEL_DIR

def setup_tensorflow_serving():
    """Setup TensorFlow Serving (Linux/Docker only)"""
    try:
        # Add tensorflow serving apt repository and update the package list
        subprocess.run([
            'bash', '-c',
            'echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list'
        ], check=True)
        
        subprocess.run([
            'bash', '-c',
            'curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -'
        ], check=True)
        
        # Update package list
        subprocess.run(['apt', 'update'], check=True)
        
        # Install tensorflow model server
        subprocess.run(['apt-get', 'install', '-y', 'tensorflow-model-server'], check=True)
        
        print("TensorFlow Serving installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error setting up TensorFlow Serving: {e}")
        print("Note: This setup requires Linux/Ubuntu environment with sudo privileges")
        return False
    except FileNotFoundError:
        print("Error: This setup requires a Linux environment (apt package manager not found)")
        return False

def start_tensorflow_server(port, model_name, model_base_path, log_file="server.log"):
    """Start TensorFlow Model Server"""
    try:
        cmd = [
            'nohup', 'tensorflow_model_server',
            f'--rest_api_port={port}',
            f'--model_name={model_name}',
            f'--model_base_path={model_base_path}'
        ]
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        print(f"Started TensorFlow Model Server on port {port}")
        print(f"Model: {model_name}")
        print(f"Logs written to: {log_file}")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Show server log
        try:
            with open(log_file, 'r') as f:
                print("Server log:")
                print(f.read())
        except FileNotFoundError:
            print("Log file not found yet")
            
        return process
        
    except FileNotFoundError:
        print("Error: tensorflow_model_server not found. Please install TensorFlow Serving first.")
        return None
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

# Deploy models (only if TensorFlow Serving setup is successful)
if setup_tensorflow_serving():
    print("\n" + "="*50)
    print("DEPLOYING MODELS")
    print("="*50)
    
    # Deploy keypoint model
    model_dir1 = deploy('/tmp/model_keypoint', model_1_facialKeyPoints)
    server1 = start_tensorflow_server(
        port=4500,
        model_name='keypoint_model',
        model_base_path=model_dir1,
        log_file='keypoint_server.log'
    )
    
    print("\n" + "-"*30)
    
    # Deploy emotion model  
    model_dir2 = deploy('/tmp/model_emotion', model_2_emo)
    server2 = start_tensorflow_server(
        port=4000,
        model_name='emotion_model', 
        model_base_path=model_dir2,
        log_file='emotion_server.log'
    )
    
    print("\n" + "="*50)
    print("DEPLOYMENT COMPLETE")
    print("="*50)
    print("Keypoint Model API: http://localhost:4500/v1/models/keypoint_model")
    print("Emotion Model API: http://localhost:4000/v1/models/emotion_model")
    
else:
    print("\n" + "="*50)
    print("ALTERNATIVE DEPLOYMENT - SAVING MODELS ONLY")
    print("="*50)
    
    # Save models in SavedModel format for manual deployment
    keypoint_dir = deploy(os.path.join(MODEL_OUTPUT_DIR, 'keypoint_savedmodel'), model_1_facialKeyPoints)
    emotion_dir = deploy(os.path.join(MODEL_OUTPUT_DIR, 'emotion_savedmodel'), model_2_emo)
    
    print(f"Models saved to:")
    print(f"Keypoint model: {keypoint_dir}")
    print(f"Emotion model: {emotion_dir}")
    print("\nTo deploy with TensorFlow Serving later, use:")
    print(f"tensorflow_model_server --rest_api_port=4500 --model_name=keypoint_model --model_base_path={keypoint_dir}")
    print(f"tensorflow_model_server --rest_api_port=4000 --model_name=emotion_model --model_base_path={emotion_dir}")


data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

# Function to make predictions from deployed models
def response(data):
    headers = {"content-type": "application/json"}
    
    # Get keypoint predictions
    json_response = requests.post('http://localhost:4500/v1/models/keypoint_model/versions/1:predict', data=data, headers=headers, verify=False)
    df_predict = json.loads(json_response.text)['predictions']
    
    # Convert data for ResNet50 emotion model (needs RGB 224x224 images)
    instances = json.loads(data)['instances']
    instances_array = np.array(instances)
    
    # Convert to RGB and resize for ResNet50
    instances_rgb = np.repeat(instances_array, 3, axis=-1)
    instances_resized = np.zeros((len(instances), 224, 224, 3), dtype=np.float32)
    for i in range(len(instances)):
        instances_resized[i] = cv2.resize(instances_rgb[i], (224, 224))
    
    # Prepare data for ResNet50 emotion model
    resnet_data = json.dumps({"signature_name": "serving_default", "instances": instances_resized.tolist()})
    
    # Get emotion predictions
    json_response = requests.post('http://localhost:4000/v1/models/emotion_model/versions/1:predict', data=resnet_data, headers=headers, verify=False)
    df_emotion = np.argmax(json.loads(json_response.text)['predictions'], axis=1)
    
    # Reshaping array from (n,) to (n,1)
    df_emotion = np.expand_dims(df_emotion, axis=1)

    # Converting the predictions into a dataframe
    df_predict = pd.DataFrame(df_predict, columns=columns)

    # Adding emotion into the predicted dataframe
    df_predict['emotion'] = df_emotion

    return df_predict

# Update data preparation for ResNet50
X_test_sample_rgb = np.repeat(X_test[0:3], 3, axis=-1)
X_test_sample_resized = np.zeros((3, 224, 224, 3), dtype=np.float32)
for i in range(3):
    X_test_sample_resized[i] = cv2.resize(X_test_sample_rgb[i], (224, 224))

data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

df_predict = response(data)

# Plotting the test images and their predicted keypoints and emotions
fig, axes = plt.subplots(3, 1, figsize = (24, 24))
axes = axes.ravel()

for i in range(3):
    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('ResNet50 Prediction = {}'.format(label_to_txt[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')


"""

Deploy the model

"""
def deploy(directory, model):
    MODEL_DIR = directory
    version = 1

    export_path = os.path.join(MODEL_DIR, str(version))
    print(f"export_path = {export_path}")

    if os.path.isdir(export_path):
        print('Already saved a model, cleaning up the folder')
        shutil.rmtree(export_path)

    tf.saved_model.save(model, export_path)

    os.environ['MODEL_DIR'] = MODEL_DIR
    return MODEL_DIR

def setup_tensorflow_serving():
    """Setup TensorFlow Serving (Linux/Docker only)"""
    try:
        # Add tensorflow serving apt repository and update the package list
        subprocess.run([
            'bash', '-c',
            'echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list'
        ], check=True)
        
        subprocess.run([
            'bash', '-c',
            'curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -'
        ], check=True)
        
        # Update package list
        subprocess.run(['apt', 'update'], check=True)
        
        # Install tensorflow model server
        subprocess.run(['apt-get', 'install', '-y', 'tensorflow-model-server'], check=True)
        
        print("TensorFlow Serving installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error setting up TensorFlow Serving: {e}")
        print("Note: This setup requires Linux/Ubuntu environment with sudo privileges")
        return False
    except FileNotFoundError:
        print("Error: This setup requires a Linux environment (apt package manager not found)")
        return False

def start_tensorflow_server(port, model_name, model_base_path, log_file="server.log"):
    """Start TensorFlow Model Server"""
    try:
        cmd = [
            'nohup', 'tensorflow_model_server',
            f'--rest_api_port={port}',
            f'--model_name={model_name}',
            f'--model_base_path={model_base_path}'
        ]
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        print(f"Started TensorFlow Model Server on port {port}")
        print(f"Model: {model_name}")
        print(f"Logs written to: {log_file}")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Show server log
        try:
            with open(log_file, 'r') as f:
                print("Server log:")
                print(f.read())
        except FileNotFoundError:
            print("Log file not found yet")
            
        return process
        
    except FileNotFoundError:
        print("Error: tensorflow_model_server not found. Please install TensorFlow Serving first.")
        return None
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

# Deploy models (only if TensorFlow Serving setup is successful)
if setup_tensorflow_serving():
    print("\n" + "="*50)
    print("DEPLOYING MODELS")
    print("="*50)
    
    # Deploy keypoint model
    model_dir1 = deploy('/tmp/model_keypoint', model_1_facialKeyPoints)
    server1 = start_tensorflow_server(
        port=4500,
        model_name='keypoint_model',
        model_base_path=model_dir1,
        log_file='keypoint_server.log'
    )
    
    print("\n" + "-"*30)
    
    # Deploy emotion model  
    model_dir2 = deploy('/tmp/model_emotion', model_2_emo)
    server2 = start_tensorflow_server(
        port=4000,
        model_name='emotion_model', 
        model_base_path=model_dir2,
        log_file='emotion_server.log'
    )
    
    print("\n" + "="*50)
    print("DEPLOYMENT COMPLETE")
    print("="*50)
    print("Keypoint Model API: http://localhost:4500/v1/models/keypoint_model")
    print("Emotion Model API: http://localhost:4000/v1/models/emotion_model")
    
else:
    print("\n" + "="*50)
    print("ALTERNATIVE DEPLOYMENT - SAVING MODELS ONLY")
    print("="*50)
    
    # Save models in SavedModel format for manual deployment
    keypoint_dir = deploy(os.path.join(MODEL_OUTPUT_DIR, 'keypoint_savedmodel'), model_1_facialKeyPoints)
    emotion_dir = deploy(os.path.join(MODEL_OUTPUT_DIR, 'emotion_savedmodel'), model_2_emo)
    
    print(f"Models saved to:")
    print(f"Keypoint model: {keypoint_dir}")
    print(f"Emotion model: {emotion_dir}")
    print("\nTo deploy with TensorFlow Serving later, use:")
    print(f"tensorflow_model_server --rest_api_port=4500 --model_name=keypoint_model --model_base_path={keypoint_dir}")
    print(f"tensorflow_model_server --rest_api_port=4000 --model_name=emotion_model --model_base_path={emotion_dir}")


data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

# Function to make predictions from deployed models
def response(data):
    headers = {"content-type": "application/json"}
    
    # Get keypoint predictions
    json_response = requests.post('http://localhost:4500/v1/models/keypoint_model/versions/1:predict', data=data, headers=headers, verify=False)
    df_predict = json.loads(json_response.text)['predictions']
    
    # Convert data for ResNet50 emotion model (needs RGB 224x224 images)
    instances = json.loads(data)['instances']
    instances_array = np.array(instances)
    
    # Convert to RGB and resize for ResNet50
    instances_rgb = np.repeat(instances_array, 3, axis=-1)
    instances_resized = np.zeros((len(instances), 224, 224, 3), dtype=np.float32)
    for i in range(len(instances)):
        instances_resized[i] = cv2.resize(instances_rgb[i], (224, 224))
    
    # Prepare data for ResNet50 emotion model
    resnet_data = json.dumps({"signature_name": "serving_default", "instances": instances_resized.tolist()})
    
    # Get emotion predictions
    json_response = requests.post('http://localhost:4000/v1/models/emotion_model/versions/1:predict', data=resnet_data, headers=headers, verify=False)
    df_emotion = np.argmax(json.loads(json_response.text)['predictions'], axis=1)
    
    # Reshaping array from (n,) to (n,1)
    df_emotion = np.expand_dims(df_emotion, axis=1)

    # Converting the predictions into a dataframe
    df_predict = pd.DataFrame(df_predict, columns=columns)

    # Adding emotion into the predicted dataframe
    df_predict['emotion'] = df_emotion

    return df_predict

# Update data preparation for ResNet50
X_test_sample_rgb = np.repeat(X_test[0:3], 3, axis=-1)
X_test_sample_resized = np.zeros((3, 224, 224, 3), dtype=np.float32)
for i in range(3):
    X_test_sample_resized[i] = cv2.resize(X_test_sample_rgb[i], (224, 224))

data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

df_predict = response(data)

# Plotting the test images and their predicted keypoints and emotions
fig, axes = plt.subplots(3, 1, figsize = (24, 24))
axes = axes.ravel()

for i in range(3):
    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('ResNet50 Prediction = {}'.format(label_to_txt[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')

    """