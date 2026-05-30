# Brain Tumor MRI Classification

A Deep Learning-based web application that classifies brain MRI scans into different tumor categories using Transfer Learning with MobileNetV2 and a Streamlit frontend.

## Live Demo

🔗 https://brain-tumor-detector-kjthtdrgroznsrjqzscmuj.streamlit.app

## Overview

This project uses a pretrained MobileNetV2 model as the feature extractor and fine-tunes the upper layers to classify brain MRI images into one of four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

Users can upload an MRI image through the Streamlit web interface and receive a prediction from the trained model.

## Features

* MRI image classification
* Transfer Learning with MobileNetV2
* Streamlit web interface
* Real-time image upload and prediction
* TensorFlow/Keras implementation
* Public deployment

## Model Architecture

The model is built using MobileNetV2 pretrained on ImageNet.

```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(4, activation="softmax")
])
```

## Technology Stack

* Python
* TensorFlow
* Keras
* MobileNetV2
* Streamlit
* NumPy
* Pandas
* Jupyter Notebook

## Project Structure

```text
brain-tumor-detector/
│
├── app.py
├── brain_tumor_classifier.keras
├── brain_tumor_training.ipynb
├── requirements.txt
└── README.md
```

## Dataset

The model was trained on a Brain MRI dataset containing images from four classes:

* Glioma
* Meningioma
* Pituitary
* No Tumor

Dataset images were resized to 224×224 before training.

## Installation

Clone the repository:

```bash
git clone https://github.com/Somsubhra-Mukherjee/brain-tumor-detector.git
cd brain-tumor-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Learning Outcomes

Through this project, I gained practical experience in:

* Transfer Learning
* Medical Image Classification
* TensorFlow/Keras workflows
* Model deployment
* Streamlit application development
* End-to-end Machine Learning pipelines

## Future Improvements

* Improve model accuracy through hyperparameter tuning
* Add Grad-CAM visualizations
* Add confidence scores
* Support multiple MRI image uploads
* Deploy using Docker and cloud infrastructure

## Contributors

* Sourasish Mukherjee
* Somsubhra Mukherjee

## Disclaimer

This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
