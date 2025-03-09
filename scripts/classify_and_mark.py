# -*- coding: utf-8 -*-
"""Classification_Marking_Final.ipynb

Developed by: Deepak Kumar
M. Tech (CSE), Amity University, Lucknow
Guide: Dr. Kapil Gupta
Assistant Professor
Department of Computer Science & Engineering,
Amity School of Engineering and Technology, Lucknow,
Amity University Uttar Pradesh, India
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from keras.preprocessing import image
import keras.backend as K

# Load Classification Model
classification_model = load_model("Classification_Basic_Model_50_epoch.h5")

# Load YOLO Detection Model
# Ensure marking_model.pt is in the same directory

detection_model = YOLO("marking_model.pt")

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_image(image_path):
    img = preprocess_image(image_path)
    prediction = classification_model.predict(img)[0][0]
    if prediction > 0.5:
        print(f"🧬 Classification: Uninfected ({prediction * 100:.2f}%)")
        return "Uninfected"
    else:
        print(f"🦠 Classification: Parasitized ({(1 - prediction) * 100:.2f}%)")
        return "Parasitized"

def detect_parasites(image_path):
    results = detection_model(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Detected Infected Areas")
    plt.show()

def process_image(image_path):
    category = classify_image(image_path)
    if category == "Parasitized":
        detect_parasites(image_path)
    else:
        print("✅ No parasites detected.")

# Example usage
image_path = "sample_image.png"  # Replace with your test image path
process_image(image_path)
