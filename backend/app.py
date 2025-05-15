from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Function to return the class name based on the prediction
def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor Detected"
    elif class_no == 1:
        return "Yes, Brain Tumor Detected"

# Function to process the image and get the prediction result
def get_result(img_path):
    image = cv2.imread(img_path)  # Read the image using OpenCV
    image = Image.fromarray(image, 'RGB')  # Convert the image to RGB format
    image = image.resize((64, 64))  # Resize the image to the required size
    image = np.array(image)  # Convert the image to a numpy array
    input_img = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    input_img = input_img / 255.0  # Normalize the image
    prediction = model.predict(input_img)  # Get the prediction from the model
    result = (prediction > 0.5).astype(int)  # Convert prediction to binary result
    return result[0][0]  # Return the result

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the home page

# Route for handling the file upload and prediction
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # Get the file from the POST request
        if not f:
            return jsonify({'error': 'No file provided'}), 400
        basepath = os.path.dirname(__file__)  # Get the base directory
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)  # Create the uploads directory if it doesn't exist
        file_path = os.path.join(upload_dir, secure_filename(f.filename))  # Create the file path
        f.save(file_path)  # Save the file
        value = get_result(file_path)  # Get the prediction result
        result = get_class_name(value)  # Get the class name based on the result
        return jsonify({'result': result})  # Return the result to the client as JSON
    return jsonify({'error': 'Invalid request'}), 400

# Run the application
if __name__ == '__main__':
    app.run(debug=True)