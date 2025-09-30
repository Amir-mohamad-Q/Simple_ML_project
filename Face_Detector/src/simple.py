import cv2 as cv
import numpy as np

# Load pre-trained classifiers from XML files for detecting faces, eyes, and smiles.
face_cascade = cv.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('../data/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('../data/haarcascade_smile.xml')

cap = cv.VideoCapture(0)

while(True):
    rec, frame = cap.read() # Read a single frame from the webcam.
    
    # Convert the color frame to grayscale for faster and more effective detection.
    frame_gr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame_gr, 1.3, 5) # Detect faces in the grayscale frame.

    # (x, y) are the starting coordinates, (w, h) are the width and height of the face rectangle.
    for (x, y, w, h) in faces:
        
        # Draw a purple rectangle around the detected face on the original color frame.
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)
        
        # Define a Region of Interest (ROI) for both the grayscale and color frames.
        # This focuses the search for eyes and smiles only within the detected face area, improving efficiency.
        frame_gr_roi = frame_gr[y:y+h, x:x+w]
        frame_roi = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(frame_gr_roi) # Detect eyes within the grayscale face ROI.

        for (ex, ey, ew, eh) in eyes:  # Loop of detected eyes.
            # Draw a green rectangle around each eye. Note this is drawn on the color face ROI.
            cv.rectangle(frame_roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        # Detect smiles within the grayscale face ROI.
        # These parameters are adjusted for smile detection, which can be less stable.
        smiles = smile_cascade.detectMultiScale(frame_gr_roi, 1.8, 20)

        # Loop of detected smiles.
        for (sx, sy, sw, sh) in smiles:
            # Draw a blue rectangle around each smile on the color face ROI.
            cv.rectangle(frame_roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    cv.imshow('frame', frame) # Display the final frame with all the rectangles in a window named 'frame'.

    keyexit = cv.waitKey(5) & 0xFF  # Close program with 'ESC' key (ASCII 27).
    if keyexit == 27:
        break

cv.destroyAllWindows() # Clean up: close all OpenCV windows.
cap.release() # Release the webcam resource.