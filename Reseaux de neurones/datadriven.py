import pandas as pd
import numpy as np
import math
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.utils import to_categorical
import tensorflow as tf
from  keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt

"""
2.1.3   -   Deep Approach - Data Augmentation 
"""

filename = 'WangSignatures.xlsx'

# Load the JCD, PHOG, CEDD, FCTH, and Fuzzy Color Histogram measurements from the excel file
jcd_df = pd.read_excel(filename, sheet_name='WangSignaturesJCD', header=None)
phog_df = pd.read_excel(filename, sheet_name='WangSignaturesPHOG', header=None)
cedd_df = pd.read_excel(filename, sheet_name='WangSignaturesCEDD', header=None)
fcth_df = pd.read_excel(filename, sheet_name='WangSignaturesFCTH', header=None)
fuzzy_df = pd.read_excel(filename, sheet_name='WangSignaturesFuzzyColorHistogr', header=None)

# Concatenate the dataframes into a single dataframe
data_df = pd.concat([jcd_df, phog_df, cedd_df, fcth_df, fuzzy_df], axis=1)

# Load the images and their corresponding labels
image_folder = 'C:/Users/cmbri/Documents/0.CNAM/Annee 3/USSI5E - Machine Learning/Machine-Learning-projects/Reseaux de neurones/Wang'

# Load dataset and resize images to the same size
images_labels = []
images = []
image_width = 256
image_height = 256
for filename in os.listdir(image_folder):
    if filename.__contains__('.jpg'):
        print(filename)
        #img = cv2.imread(os.path.join(image_folder, filename))
        #img = cv2.resize(img, (image_width, image_height)) # Resize the image to a fixed size
        #img = img.astype('float32') / 255.0 # Normalize the pixel values
        
        img = cv2.resize(img_to_array(load_img(image_folder+'/'+filename)), (image_width, image_height))
        images.append(img)
        image_label = math.ceil(int(filename.split('.')[0]) / 100)  # Extract the label from the filename
        images_labels.append(image_label)
        print(filename, image_label)

# Convert the labels to one-hot encoding
images_labels = to_categorical(images_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, images_labels, test_size=0.2, random_state=42)

X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_train = np.asarray(y_train).astype('float32') #.reshape((-1,1))
y_test = np.asarray(y_test).astype('float32') #.reshape((-1,1))

print(X_train.shape , X_test.shape, y_train.shape, y_test.shape)

# Define the model
model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='sigmoid'))


# Summary of the model
model.summary()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

plt.figure(figsize=(20, 5))
plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'valid')
plt.legend()
plt.title('Accuracy')
plt.show()

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

