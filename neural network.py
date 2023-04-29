# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

train_path = "C:\\Users\\BHARATHI\\PycharmProjects\\pythonProject1\\train"
print(train_path)

val_path = "C:\\Users\\BHARATHI\\PycharmProjects\\pythonProject1\\val"
print(val_path)

# setting the path of the variables
data_dir = train_path
img_size = (32, 32)  # resizing to 32X32

# Intializing empty list to read the input images and class names
images = []
class_val = []

#no of classes
myList = os.listdir(train_path)

# printing the folder names
print(myList)
noOfClasses = len(myList)
print("No of classes : ", len(myList))

# Reading the images from the path
for label in range(noOfClasses):
    label_path = os.path.join(data_dir, str(label))
    # Reading the images from the subfolders with class name
    for file in os.listdir(label_path):
        class_label_path = os.path.join(label_path, file)
        if class_label_path.endswith(('.tiff', '.bmp')):
            # reading the image in grayscale and scaling it
            img = cv2.imread(class_label_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            images.append(img)
            class_val.append(label)

# Convert the lists to NumPy arrays
images = np.array(images)
class_val = np.array(class_val)
# Save the arrays in NumPy format
np.save('x_train.npy', images)
np.save('y_train.npy', class_val)

# print(images, labels)

# Updating the path to validation set path
data_dir_val = val_path
img_size_val = (32, 32)
# Create empty lists for the images and labels
images_val = []
labels_val = []


# reading the images from the each class folder
for label in range(noOfClasses):
    folder_path = os.path.join(data_dir, str(label))
    for file in os.listdir(folder_path):
        class_label_path = os.path.join(folder_path, file)
        if class_label_path.endswith(('.tiff', '.bmp')):
            # reading the image in grayscale and storing them in lists
            img = cv2.imread(class_label_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            images_val.append(img)
            labels_val.append(label)


# converting the list of images and labels stored into numpy
images_val = np.array(images_val)
labels_val = np.array(labels_val)
np.save('x_test.npy', images_val)
np.save('y_test.npy', labels_val)

# the datasets to train and test
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

#printing values for testing
print('x_train', y_train, x_test, y_test)
print(f"data set samples in training : {len(x_train)}")
print(f"data set samples in test set: {len(x_test)}")
print(f"Label of image : {y_test[56]}")

# Display example images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(x_train[5], cmap='gray')
plt.title(f"Label: {y_train[6]}")
plt.subplot(1, 2, 2)
plt.imshow(x_train[56], cmap='gray')
plt.title(f"Label: {y_train[89]}")
plt.show()

# check for test set
plt.figure()
plt.imshow(x_test[45], cmap='gray')
plt.title(f"Label: {y_test[45]}")
plt.show()


# creating a simple nn with one dense layer
model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(10, input_shape=(1024,), activation='sigmoid')])
# compiling the nn
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
# training the model for some nn
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


#scaling the data sets to increase the accuracy
x_train_scaled = x_train / 255
x_test_scaled = x_test / 255
model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_test_scaled, y_test))

# evaluating test dataset
model.evaluate(x_test_scaled, y_test)


# test - 1
plt.matshow(x_test[0], cmap='gray')
y_predicted = model.predict(x_test_scaled)
print("Prediction for the first image:", np.argmax(y_predicted[0]))

# test-2
plt.matshow(x_test[88], cmap='gray')
print("Prediction for image 80:", np.argmax(y_predicted[80]))

# test - 3
plt.matshow(x_test[177], cmap='gray')
print("Prediction for image 170:", np.argmax(y_predicted[170]))


# convert to concrete values
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))

# confusion matrix computation
confusion_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
print(confusion_mat)

# confusion matrix plotting
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
plt.show()


# here we can see there are some errors
# we need to modify our nn, we add some layers in the above model and different activation function

# Adding more layers to increase accuracy
model2 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, input_shape=(1024,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# compile the nn
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy']
               )

# training the updated model
history = model2.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_test_scaled, y_test))

# evaluate test dataset on modified model
model2.evaluate(x_test_scaled, y_test)

# convert to concrete values
y_predicted = model2.predict(x_test_scaled)
y_predicted[0]
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))

#updating the confusion matrix
confusion_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
confusion_mat

# plotting the matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Observatoin : we see in the updated model, there are less number of errors,
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], color='blue')
plt.plot(history.history['val_accuracy'], color='red')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
