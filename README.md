🧠 Brain Tumor Detection Using AI
This project leverages Convolutional Neural Networks (CNNs) to detect brain tumors from MRI images, enhancing the speed and accuracy of diagnosis. Developed by Ayush Paul and Antareep Das, the model is designed to classify MRI scans as either tumor-positive or tumor-negative using deep learning techniques.

🚀 Overview
MRI images are crucial for diagnosing brain tumors, but manual interpretation is time-consuming and error-prone. Our AI model automates this process using CNNs, providing a reliable method for early detection.

Deep Learning Approach: Uses CNNs to extract hierarchical features from MRI images.

High Accuracy: Achieved a testing accuracy of 97.17% on axial MRI scans.

Real-World Impact: Aids doctors in quicker diagnosis and decision-making, especially for tumors like gliomas, meningiomas, and pituitary tumors.

🧬 Architecture
The model includes:

Convolutional Layers

MaxPooling Layers

Dropout for regularization

Dense Layers with ReLU and SoftMax activations

🛠️ Features
MRI image upload and preprocessing

Tumor detection results with classification output

Real-time prediction interface (GUI or web-based)

📊 Results
Accuracy: 97.17%

Performs best with axial brain MRI scans

SoftMax provides probabilistic output for better decision support

⚠️ Limitations
Performance may degrade with non-axial images

Requires a representative dataset for best results

🔮 Future Scope
Real-time CT/MRI diagnostics

Expansion to other cancer types (e.g., leukemia, breast cancer)

Integration of multimodal healthcare data
