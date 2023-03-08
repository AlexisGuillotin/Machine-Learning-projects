# import_measurements
import pandas as pd

# cnn_model_keras
import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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

    # Create a label vector indicating the class of each image
    label = []
    for i in range(1000):
        if i < 100:
            label.append('Jungle')
        elif i < 200:
            label.append('Plage')
        elif i < 300:
            label.append('Monument')
        elif i < 400:
            label.append('Bus')
        elif i < 500:
            label.append('Dinosaure')
        elif i < 600:
            label.append('Elephant')
        elif i < 700:
            label.append('Fleur')
        elif i < 800:
            label.append('Cheval')
        elif i < 900:
            label.append('Montagne')
        else:
            label.append('Plat')
        
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
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Reshape the data to the format expected by the CNN
        # Decomposition of test_size in prime numbers
            # test_size = 565 760
            # 565 760 = 2^9 * 5 * 13 * 17
            # 565 760 = 512 * 221 * 5
    X_train = X_train.reshape((-1, 512, 221, 5))
        # Decomposition of test_size in prime numbers
            # X_val = 141 440
            # 141 440 = 2^7 * 5 * 13 * 17
            # 141 440 = 128 * 221 * 5
    X_val = X_val.reshape((-1, 128, 221, 5))
        # Decomposition of X_test in prime numbers
            # X_val = 176 800
            # 176 800 = 2^5 * 5^2 * 13 * 17
            # 176 800 = 40 * 127 * 35
    X_test = X_test.reshape((-1, 160, 221, 5))

    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')

    # Build the CNN model
    print("\t○\tBuild the model")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 221, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print("\t○\tModel requirements")
    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

def main():
    print('•\tImportation des mesures')

    jcd_df, phog_df, cedd_df, fcth_df, fuzzy_df, data_df, label = import_measurements('WangSignatures.xlsx')

    print('•\tImportation des mesures: ok')

    print('•\tCréation du modèle CNN avec keras')
    cnn_model_keras(data_df, label)
    print('•\tCréation du modèle CNN avec keras: ok')

main()