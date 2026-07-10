import cv2
import os


def start_capture(name, image_limit=100, resolution=(640, 480)):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("Directory already exists")

    # Start video capture and set resolution
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error: Could not open video stream.")
        return
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    while True:
        ret, img = vid.read()
        if not ret:
            print("Failed to capture video stream")
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"Face Detected | {num_of_images} images captured",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Crop and save the detected face
            face_img = gray_img[y : y + h, x : x + w]  # Save grayscale for smaller size
            file_path = os.path.join(path, f"{num_of_images}_{name}.jpg")

            try:
                cv2.imwrite(file_path, face_img)
                num_of_images += 1
                # print(f"Saved: {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")

        # Display the frame with rectangle and text
        cv2.imshow("Face Detection", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27 or num_of_images >= image_limit:
            break

    vid.release()
    cv2.destroyAllWindows()
    return num_of_images
