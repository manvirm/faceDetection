# Read images, videos, etc
import cv2

# Load pre-trained data on face frontals from opencv (haar cascade algo)
# Classifiers are detectors (face detector in this case)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces (imread = image read)
# Image is just an array (pixels = numbers)
img = cv2.imread('kreeves.jpg')

# Show image
cv2.imshow('Programming Face Detector', img)

# Pause code so image will show until key is pressed
cv2.waitKey()

print("code completed")