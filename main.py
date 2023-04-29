# Installed all the packages required
# import libraries
import keras.engine.sequential
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import *

# path of the train and val folders
path = 'train'
path2 = 'val'
myList = os.listdir(path)
myList2 = os.listdir(path2)
# printing the folder names
print(myList)
print("validation images : ", myList2)
print("Total no of classes : ", len(myList))
noOfClasses = len(myList)

images = []
valImg = []
classNo = []
valCLassNo = []
# iterating to read images from each class
print("Reading the classes for tarining set")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        currentImg = cv2.imread(path + "/" + str(x) + "/" + y)
        currentImg = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)
        currentImg = cv2.resize(currentImg, (32, 32))  # resizing as its computationally expensive
        images.append(currentImg)
        classNo.append(x)
    print(x, end=" ")
print("\n Total no of images readed:", len(images))

# loop for reading validation sets
print("Reading the classes for validation ")
for a in range(0, noOfClasses):
    myPicList2 = os.listdir(path2 + "/" + str(a))
    for b in myPicList2:
        cValImg = cv2.imread(path2 + "/" + str(a) + "/" + b)
        cValImg = cv2.cvtColor(cValImg, cv2.COLOR_BGR2GRAY)
        cValImg = cv2.resize(cValImg, (32, 32))
        valImg.append(cValImg)
        valCLassNo.append(a)
    print(a, end=" ")
print("\n Total no of images readed for validation:", len(valImg))

# converting the list to numpy array
images = np.array(images)
classNo = np.array(classNo)
valImg = np.array(valImg)
valCLassNo = np.array(valCLassNo)
# print("checking images and class shape : ", images.shape, classNo.shape)
# cv2.imshow("window1", valImg[10])

# splitting the datasets for train and test
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
# print(X_train.shape, y_train.shape)
X_val, y_val = valImg, valCLassNo
# print(valImg.shape, valCLassNo.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
# print(numOfSamples)

# checking the distribution of the samples
plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of samples per each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images/samples ")
plt.show()

# def preProcessing(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
#     img = img/255
#     return img


# preprocessing all the images
# print(X_train, X_test, X_val)
# for i in X_train:
#     X_train = np.array(preProcessing(X_train[i]))

# X_train = np.array(map(preProcessing, X_train))
# print("X_train values ", X_train.shape)
# X_test = np.array(map(preProcessing, X_test))
# X_val = np.array(map(preProcessing, X_val))
# print("preprocessing done ", X_train, X_test, X_val)

# cv2.imshow("window1", X_train[10])
# img = preProcessing(X_train[10])
# #img = cv2.resize(img, (300, 300))
# cv2.imshow("preprocessing", img)
# cv2.waitKey(0)

# reshaping to add depth to images
# print("in reshaping ...", X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# print("after Preprocessing :", X_val.shape)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# code
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)


def myModel():
    # model = keras.Sequential()
    # model.add(keras.layers.Dense(10, input_shape=(32, 32, 1), activation='relu'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(10, input_shape=(32,), activation='sigmoid')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

batchSizeVal = 10
epochVal = 10
stepsPerEpochVal = 2000
model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal), steps_per_epoch=stepsPerEpochVal,
                    epochs=epochVal, validation_data=(X_test, y_test), shuffle=1)

# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
# model.evaluate(X_test, y_test)


