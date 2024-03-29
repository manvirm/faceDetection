# Read images, videos, etc
import cv2
import random

# Load pre-trained data on face frontals from opencv (haar cascade algo)
# Classifiers are detectors (face detector in this case)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces (imread = image read)
# Image is just an array (pixels = numbers)
#img = cv2.imread('kreeves.jpg')

# Use webcam, '0' will use default webcam
# Can also use videos for this
webcam = cv2.VideoCapture(0)

# Iterate over frames until video ends (webcam ends)

while True:

    # Read frames, first var is boolean, and second one is frame
    # Use frame instead of image
    successful_frame_read, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # No matter size of face it will detect (multi scale)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)


    for (x, y, w, h) in face_coordinates:

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5)

    # Show Frame
    cv2.imshow('Programming Face Detector', frame)

    # Pause code so image will show until key is pressed
    # Instead of waiting for key, we wait 1ms, so wont pause until key is pressed
    # Wait is needed so popup will stay
    key = cv2.waitKey(1)

    # If Q is pressed, stop loop (ASCII CODE FOR Q)
    if key == 81 or key == 113:
        break

# Clean up code
webcam.release()

'''
# Choose an image to detect faces (imread = image read)
# Image is just an array (pixels = numbers)
img = cv2.imread('kreeves.jpg')


# Must convert image to grayscale for algorithm to recognize
# This function converts image to specific colors
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# No matter size of face it will detect (multi scale)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Format: [[top left point, width, height]]  
#print(face_coordinates)

# Draw rectangles around the face
# First 2 tuples are 1.top left point and 2.bottom right point of rectangle
# 3rd tuple is color of rectangle (blue RGB in this case)
# Last argument is the thickness of rectangle
#cv2.rectangle(img, (top left), (222+width, 292+height),  (0, 0, 255), 2)
#cv2.rectangle(img, (222, 292), (222+340, 292+340),  (0, 0, 255), 2)

for (x, y, w, h) in face_coordinates:

    r1 = random.randint(0, 255)
    r2 = random.randint(0, 255)
    r3 = random.randint(0, 255)
    cv2.rectangle(img, (x,y), (x+w, y+h), (r1, r2, r3), 5)

# Show image
cv2.imshow('Programming Face Detector', img)

# Pause code so image will show until key is pressed
cv2.waitKey()
'''