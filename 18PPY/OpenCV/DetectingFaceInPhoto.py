import cv2
#absolute path to cascade file
face_cascade = cv2.CascadeClassifier('Documents/18PPY/OpenCV/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml')
#absolute path to photo file
img = cv2.imread('Documents/18PPY/OpenCV/MoreThanOne.jpg')

grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayscaleImage, 1.1, 4)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h),(64, 245, 61),2)

cv2.imshow('Detecting face in photo',img)

cv2.waitKey()