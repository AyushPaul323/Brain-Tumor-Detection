import cv2 
import os
from PIL import Image
from tensorflow import keras 
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Directory containing the dataset
image_directory = 'datasets/'

# Lists of image filenames for both classes
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

# Lists to hold the dataset and labels
dataset = []
label = []

# Define the input size for the images
INPUT_SIZE = 64

# Load images with no tumor, resize, and add to dataset with label 0
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):  # Check if the file is a jpg
        image = cv2.imread(image_directory + 'no/' + image_name)  # Read the image
        image = Image.fromarray(image, 'RGB')  # Convert to PIL Image
        image = image.resize((INPUT_SIZE, INPUT_SIZE))  # Resize the image
        dataset.append(np.array(image))  # Add image to dataset
        label.append(0)  # Add label 0 for no tumor

# Load images with tumor, resize, and add to dataset with label 1
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):  # Check if the file is a jpg
        image = cv2.imread(image_directory + 'yes/' + image_name)  # Read the image
        image = Image.fromarray(image, 'RGB')  # Convert to PIL Image
        image = image.resize((INPUT_SIZE, INPUT_SIZE))  # Resize the image
        dataset.append(np.array(image))  # Add image to dataset
        label.append(1)  # Add label 1 for tumor

# Convert dataset and labels to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize data
x_train = normalize(x_train, axis=1)  # Normalize the training data
x_test = normalize(x_test, axis=1)  # Normalize the testing data

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the model
model = Sequential()

# Add first convolutional layer with ReLU activation and max pooling
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolutional layer with ReLU activation and max pooling
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add third convolutional layer with ReLU activation and max pooling
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add a dense layer with ReLU activation
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))  # Add dropout for regularization

# Add the output layer with softmax activation
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = score[1] * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the trained model
model.save('BrainTumor10EpochsCategorical.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()