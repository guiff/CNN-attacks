import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


classesNumber = 26
batchSize = 128
epochs = 20
imgX = 16
imgY = 8


def getData(fileName):
    data = pd.read_csv(fileName)
    labels = oneHotEncoding(data.as_matrix(columns=['Prediction'])) # One hot encoding of the labels
    dataset = data.drop(['Prediction','Id',"NextId","Position"], axis=1).as_matrix()
    dataset = dataset.reshape(dataset.shape[0], imgX, imgY, 1).astype('float32') # Reshape the dataset into a 4D tensor
    return labels, dataset


def oneHotEncoding(labels):
    for i in range(labels.shape[0]):
        labels[i][0] = ord(labels[i][0]) - ord('a') # Convert letters to numbers
    return np_utils.to_categorical(labels, classesNumber) # One hot encoding of the labels


# Define the convolutional neural network model
def createModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=(imgX, imgY, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(classesNumber, activation='softmax'))
    return model


train_labels, train_dataset = getData("train.csv")

model = createModel()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model
model.fit(train_dataset, train_labels, epochs=epochs, batch_size=batchSize, verbose=2) # Fit the model
model.save('model.h5') # Save the model
