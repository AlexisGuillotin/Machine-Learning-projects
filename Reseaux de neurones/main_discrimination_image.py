# import_measurements
import pandas as pd

# cnn_model_keras
import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def import_measurements(filename):
    # Read the JCD measurements from the excel file
    jcd_df = pd.read_excel(filename, sheet_name='WangSignaturesJCD', header=None)

    # Read the PHOG measurements from the excel file
    phog_df = pd.read_excel(filename, sheet_name='WangSignaturesPHOG', header=None)

    # Read the CEDD measurements from the excel file
    cedd_df = pd.read_excel(filename, sheet_name='WangSignaturesCEDD', header=None)

    # Read the FCTH measurements from the excel file
    fcth_df = pd.read_excel(filename, sheet_name='WangSignaturesFCTH', header=None)

    # Read the Fuzzy Color Histogram measurements from the excel file
    fuzzy_df = pd.read_excel(filename, sheet_name='WangSignaturesFuzzyColorHistogr', header=None)

    # Concatenate the dataframes into a single dataframe
    data_df = pd.concat([jcd_df, phog_df, cedd_df, fcth_df, fuzzy_df], axis=1)

    # create label vector
    label = np.zeros(1000)
    for i in range(10):
        label[i*100:(i+1)*100] = i

    return jcd_df, phog_df, cedd_df, fcth_df, fuzzy_df, data_df, label

def cnn_model_keras(data_df, label):
    # Delete target column from dataframe
    del data_df[0]

    # Convert the data dataframe and label vector to numpy arrays
    X = data_df.values
    y = np.array(label)

    # Convert the label vector to one-hot encoding
    y = to_categorical(y)

    # Split the dataset into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
     
    # Build the CNN model
    print("\t○\tBuild the model")
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    print("\t○\tModel compilation")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
 
    # Train the model
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert the one-hot encoded vector back to labels
    y_pred = np.argmax(y_pred, axis=1)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc, ", Test loss: ", test_loss)

        # Plot the confusion matrix
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='Oranges')
    plt.show()

    return model


def main():

    print('•\tImportation des mesures')
    jcd_df, phog_df, cedd_df, fcth_df, fuzzy_df, data_df, label = import_measurements('WangSignatures.xlsx')
    print('•\tImportation des mesures: ok')

    print('•\tCréation du modèle CNN avec keras')
    cnn_model_keras(data_df, label)
    print('•\tCréation du modèle CNN avec keras: ok')

main()