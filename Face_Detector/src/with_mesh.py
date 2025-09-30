import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector  # cvzone library for easy face detection
from cvzone.FaceMeshModule import FaceMeshDetector  # cvzone library for creating a face mesh


cap = cv.VideoCapture(1) # Initialize the webcam. '1' usually refers to an external camera. Use '0' for the default built-in camera.
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)


while True: # a loop to continuously capture frames from the webcam.

    # Read a single frame from the webcam.
    rec, frame = cap.read() # 'rec' is a boolean that is True if the frame is read successfully, 'frame' is the captured image.

    # Find faces in the current frame. 
    frame, bbox = detector.findFaces(frame) # This returns the frame with bounding boxes drawn and a list 'bbox' with face information.

    # Find the face mesh on the detected face.
    frame, faces = meshdetector.findFaceMesh(frame) # This returns the frame with the mesh drawn on it and a list 'faces' containing mesh data.

    # Check if any faces were detected (if bbox list is not empty).
    if bbox:
        center = bbox[0]["center"]  # Get the center coordinates of the first detected face's bounding box.
        # This line is commented out. If you uncomment it, it will draw a filled purple circle at the center of the face.
        cv.circle(frame, center, 5, (255, 0, 255), cv.FILLED)

    cv.imshow('frame', frame) # Display the processed frame in a window named 'frame'.

    # Wait for 5 milliseconds for a key press and get the key code.
    keyexit = cv.waitKey(5) & 0xFF

    # If the key pressed is the 'Escape' key (ASCII value 27), break the loop.
    if keyexit == 27:
        break


cap.release() # Release the webcam resource.
cv.destroyAllWindows() # Close all the OpenCV windows that were created.