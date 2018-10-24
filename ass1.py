from __future__ import print_function
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from matplotlib import pyplot
import psutil

from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

batch_size = 300
# specify the number of epochs 
nb_epoch = 2

# create a grid of 3x3 images 
for i in range(0, 9): 
    pyplot.subplot(330 + 1 + i) 
    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
# show the plot
pyplot.show()

# 4. Load pre-shuffled MNIST data into train and test sets
print('Training data shape : ', X_train.shape)

print('Testing data shape : ', X_test.shape)

plt.figure(figsize=[4,2])

# Display the first image in training data
plt.subplot(221)
plt.imshow(X_train[0], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))

# Display the first image in testing data
plt.subplot(222)
plt.imshow( X_test[0], cmap='gray')
plt.title("Ground Truth : {}".format( y_test[0]))

# Find the shape of input images and create the variable input_shape
#nRows,nCols,nDims = X_train.shape[1:]
#train_data = X_train.reshape(X_train.shape[0], nRows, nCols, nDims)
#test_data = X_test.reshape(X_test.shape[0], nRows, nCols, nDims)
#input_shape = (nRows, nCols, nDims)

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
# Form the CNN
def createModel():
    # Layer one (every layer has a filter size here it is 16)
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(1,28,28)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    # Layer two (every layer has a filter size here it is 32)
    model.add(Conv2D(32, (3, 3), padding='same' ,activation='relu'))
    model.add(Conv2D(32,( 3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # Layer three (every layer has a filter size here it is 64)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())

    # Consists of a hidden layer with 50 nodes
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))

    # Gives 10 outputs
    model.add(Dense(10, activation='softmax'))

    return model
model1 = createModel()

# 8. Compile model
#gradiant decent algorith used is adam gradiant decent algorith, with 0.001 learning rate
model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 9. Fit model on training data
history =model1.fit(X_train, Y_train, validation_data=(X_test, Y_test),batch_size=batch_size, epochs=nb_epoch)

# 10. Evaluate model on test data
score = model1.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])

# Loss Curves
plt.figure(2,figsize=[8,6])
plt.plot(range(nb_epoch),history.history['loss'],'r',linewidth=3.0)
plt.plot(range(nb_epoch),history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('Loss.png')

# Accuracy Curves
plt.figure(2,figsize=[8,6])
plt.plot(range(nb_epoch),history.history['acc'],'r',linewidth=3.0)
plt.plot(range(nb_epoch),history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('accuracy.png')
