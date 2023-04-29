#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow import keras

train_path = r"C:\\Users\\BHARATHI\\PycharmProjects\\pythonProject1\\train"
train_path

val_path = r"C:\\Users\\BHARATHI\\PycharmProjects\\pythonProject1\\val"
val_path

# Set the path to the folder containing the 'train' folder
data_directory = train_path
# Set the image size
img_size = (32, 32)
# Create empty lists for the images and labels
images = []
labels = []


# In[5]:


# Loop over each folder from '0' to '9'
for label in range(10):
 	folder_path = os.path.join(data_directory, 'train', str(label))
 	# Loop over each image in the folder
 	for file in os.listdir(folder_path):
 		file_path = os.path.join(folder_path, file)
 		if file_path.endswith(('.tiff','.bmp')):
	 		# Load the image and resize it to the desired size
	 		img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	 		img = cv2.resize(img, img_size)
			 # Append the image and label to the lists
 			images.append(img)
 			labels.append(label)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
# Save the arrays in NumPy format
np.save('x_train.npy', images)
np.save('y_train.npy', labels)


print(images,labels)


# Set the path to the folder containing the 'val' folder
data_dir_val = val_path
# Set the image size
img_size_val = (32, 32)
# Create empty lists for the images and labels
images_values = []
labels_values = []


# In[9]:


# Loop over each folder from '0' to '9'
for label in range(10):
    folder_path=os.path.join(data_directory,'val',str(label))
    #loop over each image in the folder
    for file in os.listdir(folder_path):
        file_path=os.path.join(folder_path,file)
        if file_path.endswith(('.tiff','.bmp')):
            # Load the image and resize it to the desired size
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            #append the image and label to the lists
            images_values.append(img)
            labels_values.append(label)


# In[11]:


# Convert the lists to NumPy arrays
images_values = np.array(images_values)
labels_values = np.array(labels_values)
# Save the arrays in NumPy format
np.save('x_test.npy', images_values)
np.save('y_test.npy', labels_values)


# In[12]:


# Load the dataset
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')


# In[13]:


print('x_train',y_train,x_test,y_test)


# In[22]:


# test the images are loaded correctly
print(f"Number of images in training set: {len(x_train)}")
print(f"Number of images in test set: {len(x_test)}")
print(f"Shape of first training image: {x_train[0].shape}")
print(f"Label of first training image: {y_train[0]}")
print(f"Label of 130th test image: {y_test[130]}")

# Display example images
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.imshow(x_train[1], cmap='gray')
plt.title(f"Label: {y_train[1]}")
plt.subplot(1, 2, 2)
plt.imshow(x_train[988], cmap='gray')
plt.title(f"Label: {y_train[972]}")
plt.show()

# Display an image from the test set
plt.figure()
plt.imshow(x_test[120], cmap='gray')
plt.title(f"Label: {y_test[122]}")
plt.show()


# In[25]:


# creating a simple nn
# create a dense layer where every input is connected to every other output, the number of inputs are 1000, outputs are 10
# activation function is sigmoid
model = keras.Sequential([
 keras.layers.Flatten(),keras.layers.Dense(10, input_shape=(1024,),activation = 'sigmoid')
])
# compile the nn
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
# train the model
# some 10 iterations done here
model.fit(x_train, y_train,epochs= 10, validation_data=(x_test, y_test))


# In[ ]:



# Observation : we see a better accuracy from the 2nd iteration
# now scale and try to check the accuracy, divide dataset by 255


# In[26]:



x_train_scaled = x_train/255
x_test_scaled = x_test/255
model.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# In[ ]:


# Observation : we got better result for all iterations on scaling the training dataset


# In[27]:


# evaluate test dataset
model.evaluate(x_test_scaled,y_test)


# In[34]:


# Predict the first image
plt.matshow(x_test[0], cmap='gray')
y_predicted = model.predict(x_test_scaled)
print("Prediction for the first image:", np.argmax(y_predicted[0]))

# Test some more values
plt.matshow(x_test[88], cmap='gray')
print("Prediction for image 78:", np.argmax(y_predicted[78]))

plt.matshow(x_test[177], cmap='gray')
print("Prediction for image 144:", np.argmax(y_predicted[144]))


# In[35]:



# some predictions may not be not right
# build confusion matrix to see how our prediction looks like
# convert to concrete values
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat


# In[40]:


import seaborn as sns

plt.figure(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
plt.show()


# In[ ]:


# here we can see there are some errors
# we need to modify our nn, we add some layers in the above model and different activation function


# In[41]:



# in 1st Dense layer,the input is 32 x 32 = 1024 neurons, which will give 10 output(numbers from 0 to 9)
# 2nd Dense layer,the input is 10 neurons from above layers output
# we can add more layers for accuracy
model2 = keras.Sequential([
 keras.layers.Flatten(),
 keras.layers.Dense(1024,input_shape=(1024,), activation='relu'),
 keras.layers.Dense(10, activation='softmax')
])
# compile the nn
model2.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
# train the model
# some 10 iterations done here
history = model2.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# In[42]:


# Observation : due to multiple layers the compiling will take more time to execute
# we also got amazing accuracy than earlier
# evaluate test dataset on modified model
model2.evaluate(x_test_scaled,y_test)


# In[43]:


# Earlier we got 0.9213483333587646 now we got 0.9606741666793823 accuracy
# redo the confusion matrix
# build confusion matrix to see how our prediction looks like
# convert to concrete values
y_predicted = model2.predict(x_test_scaled)
y_predicted[0]
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat


# In[45]:


plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[ ]:


# Observatoin : we see in the updated model, there are less number of errors,


# In[47]:


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







