import cv2
from time import sleep
from PIL import Image
import os


def load_classifier(name):
    """Load face recognizer and haarcascade classifier."""
    face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")

    if not os.path.exists(f"./data/classifiers/{name}_classifier.xml"):
        raise FileNotFoundError(f"Classifier for {name} not found.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"./data/classifiers/{name}_classifier.xml")
    return face_cascade, recognizer


def resize_image(input_path, output_path, dim=(124, 124)):
    """Resize image and save it to the output path."""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_path}.")

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)


def overlay_images(
    base_img_path, overlay_img_path, position=(195, 114), output_path="end.png"
):
    """Overlay one image on top of another and save the result."""
    base_img = Image.open(base_img_path).copy()
    overlay_img = Image.open(overlay_img_path).copy()

    base_img.paste(overlay_img, position)
    base_img.save(output_path)


def recognize_face(frame, gray_frame, face_cascade, recognizer, name):
    """Detect and recognize the face in the frame."""
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    pred = 0

    for x, y, w, h in faces:
        roi_gray = gray_frame[y : y + h, x : x + w]

        id, confidence = recognizer.predict(roi_gray)
        confidence_level = 100 - int(confidence)
        font = cv2.FONT_HERSHEY_PLAIN

        if confidence_level > 50:
            pred += 1
            label = name.upper()
            color = (0, 255, 0)  # Green for recognized face
        else:
            pred -= 1
            label = "UnknownFace"
            color = (0, 0, 255)  # Red for unknown face

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = cv2.putText(frame, label, (x, y - 4), font, 1, color, 1, cv2.LINE_AA)

    return frame, pred


def process_recognition(name, pred):
    """Handle actions after face recognition."""
    if pred > 0:
        dim = (124, 124)
        input_img_path = f".\\data\\{name}\\{pred}_{name}.jpg"
        output_img_path = f".\\data\\{name}\\50{name}.jpg"

        resize_image(input_img_path, output_img_path)
        overlay_images(
            ".\\assets\\verify.png",
            output_img_path,
            (195, 114),
            ".\\assets\\cache\\end.png",
        )

        # Display final result
        result_frame = cv2.imread("./assets/cache/end.png", 1)
        if result_frame is not None:
            cv2.imshow("Result", result_frame)
            cv2.waitKey(5000)


def main_app(name):
    try:
        face_cascade, recognizer = load_classifier(name)
    except FileNotFoundError as e:
        print(e)
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    pred = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame, pred = recognize_face(frame, gray_frame, face_cascade, recognizer, name)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            print(f"Prediction Score: {pred}")
            process_recognition(name, pred)
            break

    cap.release()
    cv2.destroyAllWindows()
