import cv2
import numpy as np
import requests
from io import BytesIO

# Display menu for user to choose an option
print("Choose an option:")
print("1. Detect faces from webcam")
print("2. Detect faces from photo 1")
print("3. Detect faces from photo 2")
print("4. Detect faces from photo 3")
print("5. Detect faces from photo 4")
choice = input("Enter your choice (1, 2, 3, 4, or 5): ")

if choice == '1':
    # Create a classifier for detecting faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open a video stream from the webcam
    cap = cv2.VideoCapture(0)  # 0 indicates the first webcam, you can change it to other values based on the number of connected webcams

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (64, 245, 61), 2)

        cv2.imshow('Detecting faces in webcam', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # Pressing the "ESC" key will close the program
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice in ['2', '3', '4', '5']:
    # Download face cascade classifier from the website
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    response = requests.get(cascade_url)
    xml_data = response.content

    # URLs for the photos
    image_urls = [
        "https://github.com/AndriiKruhliak/Python/blob/main/18PPY/OpenCV/photos/CoupleOfFaces.jpg?raw=true",
        "https://github.com/AndriiKruhliak/Python/blob/main/18PPY/OpenCV/photos/dog.jpg?raw=true",
        "https://github.com/AndriiKruhliak/Python/blob/main/18PPY/OpenCV/photos/photo.jpg?raw=true",
        "https://github.com/AndriiKruhliak/Python/blob/main/18PPY/OpenCV/photos/test.png?raw=true"
    ]
    
    image_url = image_urls[int(choice) - 2]  # Select the appropriate URL based on user's choice
    photo_response = requests.get(image_url)
    image_data = photo_response.content

    image_bytes = BytesIO(image_data)

    # Read the image from byte array
    img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(grayscaleImage, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (64, 245, 61), 2)

    # Show the image with rectangles around the faces
    cv2.imshow('Detecting faces in photo', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
