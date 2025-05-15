import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')

# Read and preprocess the image
image = cv2.imread('C:\\Final Year Project ML\\Brain Tumour\\pred\\pred2.jpg')  # Read the image using OpenCV

img = Image.fromarray(image)  # Convert the image to a PIL Image object
img = img.resize((64, 64))  # Resize the image to the required size (64x64 pixels)
img = np.array(img)  # Convert the image to a numpy array

# Reshape the image to match the input shape of the model
input_img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input shape

# Normalize the image
input_img = input_img / 255.0  # Normalize the pixel values to the range [0, 1]

# Predict the class
prediction = model.predict(input_img)  # Get the prediction from the model

# Print the prediction probabilities
print("Prediction Probabilities:", prediction)

# Apply a threshold to get the class label
result = (prediction > 0.5).astype(int)  # Convert prediction to binary result

print("Prediction Result:", result)  # Print the result

# Plot the prediction probabilities
classes = ['No Tumor', 'Yes Tumor']
plt.bar(classes, prediction[0])
plt.title('Prediction Probabilities')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.show()