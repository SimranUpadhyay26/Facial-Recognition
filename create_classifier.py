import numpy as np
from PIL import Image
import os
import cv2
import re


def train_classifer(name, target_size=(200, 200), batch_size=32):
    path = os.path.join(os.getcwd(), "data", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist!")

    faces, ids = [], []

    def preprocess_image(image_path):
        try:
            img = Image.open(image_path).convert("L")  # Convert to grayscale
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, "uint8")

            # Optional: histogram equalization to improve contrast
            img_array = cv2.equalizeHist(img_array)
            return img_array
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    # Traverse through all images in the dataset directory
    for root, dirs, files in os.walk(path):
        pictures = files

    if not pictures:
        raise ValueError(f"No images found in the directory {path}!")

    # Preprocess each image
    for pic in pictures:
        imgpath = os.path.join(path, pic)
        img_array = preprocess_image(imgpath)
        if img_array is not None:
            match = re.match(r"^(\d+)_", pic)
            if match:
                id = int(match.group(1))  # Extract the number
                faces.append(img_array)
                ids.append(id)

    if not faces:
        raise ValueError("No valid faces were processed!")

    ids = np.array(ids)

    # Initialize the LBPH face recognizer
    clf = cv2.face.LBPHFaceRecognizer_create()

    num_images = len(faces)
    num_batches = num_images // batch_size

    # Train the classifier in batches
    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        print(f"Training batch {i + 1}/{num_batches}...")
        clf.update(faces[start:end], ids[start:end])

    # Train the remaining data
    remaining_faces = faces[num_batches * batch_size :]
    remaining_ids = ids[num_batches * batch_size :]
    if remaining_faces:
        print("Training remaining images...")
        clf.update(remaining_faces, remaining_ids)

    # Save the trained classifier
    classifier_path = os.path.join("data", "classifiers", f"{name}_classifier.xml")
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
    clf.write(classifier_path)
    print(f"Classifier saved at {classifier_path}")

    return classifier_path
