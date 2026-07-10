import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load models and files
face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
emotion_model = "./data/_mini_XCEPTION.106-0.65.hdf5"
emotion_classifier = load_model(emotion_model, compile=False)
ageProto = "./data/age_deploy.prototxt"
ageModel = "./data/age_net.caffemodel"
genderProto = "./data/gender_deploy.prototxt"
genderModel = "./data/gender_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Constants and Labels
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
genderList = ["Male", "Female"]
Emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


# Function to detect and predict age, gender, and emotion
def detect_and_predict(frame):
    default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        roi_rgb = default_img[y : y + h, x : x + w]
        roi_gray = gray_img[y : y + h, x : x + w]

        # Preprocessing for age and gender
        blob = cv2.dnn.blobFromImage(
            roi_rgb, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )

        # Predict Gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Emotion Prediction
        roi_gray_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_gray_resized = img_to_array(roi_gray_resized)
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)

        preds = emotion_classifier.predict(roi_gray_resized)[0]
        emotion = Emotions[np.argmax(preds)]

        # Draw rectangle and put predictions on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{gender}, {age}, {emotion}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )


# Main function
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        detect_and_predict(frame)

        # Show the result in a window
        cv2.imshow("Age, Gender, and Emotion Prediction", frame)

        # Press 'q' or ESC to exit
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
