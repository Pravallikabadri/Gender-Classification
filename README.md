# Gender-Classification
<img src="gender detection (2).jpeg" alt="Gender Detection Sample Output" width="800" height="400">>

# Project Title  
### Gender detection: Gender Classification Using Deep learning
# Description
Gender Classification Using Deep Learning and OpenCV is a real-time facial gender detection project powered by a custom-built Convolutional Neural Network (CNN). Trained on grayscale facial images resized to 32x32 pixels, the model classifies detected faces as male or female with reliable accuracy.

The project utilizes OpenCV's Haar cascade classifier for face detection and Keras for building and training the CNN. It incorporates preprocessing techniques like normalization and image resizing, and leverages train_test_split for model evaluation during training using categorical crossentropy and the Adam optimizer.

Streamlit is used to create an interactive interface, allowing users to upload images or use a webcam for real-time gender detection. The model displays bounding boxes with labels over each detected face and outputs gender counts, making it suitable for lightweight real-world applications like demographic analytics and user interaction systems.
# Table of contents
- [Project Overview](#project-Overview)
- [Dataset](#datasets)
- [Dependencies](#dependencies)
- [Required Imports & Libraries](#required-imports-Libraries)
- [Project Structure](#project-Structure)
- [Data Preprocessing](#data-Preprocessing)
- [Model Architecture](#model-architecture)
- [Training Details](#training-Details)
- [Performance Metrics](#performance-Metrics)
- [Running the App](#running-the-App)
- [Sample Output](#sample-Output)
- [Future Work](#future-Work)
# Project Overview
This project detects the gender of individuals from facial images using a CNN-based model trained on grayscale images. It leverages OpenCV for face detection, Keras for deep learning, and Streamlit to provide an interactive user interface.

# Dataset
Dataset source:


## Dependencies
Before running the project, ensure the following Python libraries are installed:

- `tensorflow / keras`: For building, training, and loading the CNN model.

- `opencv-python`: For face detection using Haar cascades and image processing.

- `streamlit`: To create the interactive web interface for image and webcam input.

- `numpy`: For numerical operations and handling image arrays.

- `Pillow`: For image handling and conversion in the Streamlit app.

- `scikit-learn`: For dataset splitting and evaluation.

To install these dependencies, run the following command:

```sh
pip install tensorflow keras opencv-python streamlit numpy Pillow scikit-learn
```
## Required Imports and Libraries
```sh
import numpy as np
import cv2
import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from PIL import Image
```
## Project Structure
```sh
gender-detection/
│
├── genderdetector.py              # Streamlit web app for detection
├── mainTrain01.py                 # Model training script
├── trainingDataTarget/
│   ├── data.npy                   # Numpy array of training images
│   ├── target.npy                 # Numpy array of labels
│   ├── model-019.model            # Saved trained model
│   └── haarcascade_frontalface_default.xml
├── README.md                      # Project documentation
└── requirements.txt               # Dependency list
```
## Dara Preprocessing
### Training Data Preparation:
```sh
data = np.load('./trainingDataTarget/data.npy')
target = np.load('./trainingDataTarget/target.npy')

train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.1
)
```
### Image Normalization & Reshaping (During Prediction):
```sh
resized = cv2.resize(sub_face_img, (32, 32))
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 32, 32, 1))
```
## Model Architectures Used
We used a custom Convolutional Neural Network (CNN) architecture tailored for binary gender classification from facial images:

- `Conv2D (32 filters)`: The model begins with two convolutional layers, each with 32 filters of size 3×3 and ReLU activation, to extract low-level features like edges and textures.

- `MaxPooling2D`: A pooling layer reduces the spatial dimensions to downsample feature maps and control overfitting.

- `Conv2D (64 filters)`: Followed by two deeper convolutional layers with 64 filters to extract more complex patterns and facial structures relevant to gender.

- `Dropout (0.5)`: Dropout layers are used after convolution and dense blocks to prevent overfitting during training.

- `Flatten + Dense`: The feature maps are flattened and passed through a fully connected dense layer with 64 neurons and ReLU activation.

- `Output Layer (Dense with Softmax)`: The final dense layer has 2 output units (for Male and Female) with softmax activation to produce class probabilities.

This lightweight custom CNN is optimized for fast and effective real-time gender classification on low-resolution 32x32 grayscale facial images.

## 🎯Training Details

- Optimizer: Adam

- Loss Function: Categorical Crossentropy

- Epochs: 20

- Validation Split: 20%

- Checkpoint: Saves best model by validation loss

- Model Save Format: model-xxx.model per epoch
  
## 📊Performance Metrics
Best Model: Custom CNN (saved as model-019.model based on lowest validation loss)

Metrics Used: Accuracy (during training and validation)

Plots: Not included in the current code, but model accuracy and loss are tracked via Keras callbacks during training.
`loss`
<img src="cnn_training_plot.png" alt="CNN Accuracy and Loss Curves" width="800" height="400">





