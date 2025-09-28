# Real-Time Face and Mesh Detection

This project uses your webcam to perform real-time face detection and facial landmark tracking. It overlays a bounding box on detected faces and a detailed 468-point face mesh.

## Features

*   **Real-time Face Detection:** Quickly identifies and locates human faces in a video stream.
*   **Facial Mesh Tracking:** Generates a detailed 3D mesh of facial landmarks.
*   **Simple Visualization:** Overlays the bounding box and mesh directly onto the camera feed.
*   **Lightweight and Efficient:** Built with the efficient `cvzone` library for smooth real-time performance.

## Requirements

Before you begin, ensure you have Python installed on your system. This script relies on the following Python libraries:

*   **OpenCV:** For accessing the camera and handling image processing.
*   **cvzone:** A computer vision library that simplifies tasks like face and hand tracking.
*   **MediaPipe:** A powerful framework by Google used by `cvzone` for its detection models.

## Installation

1.  **Clone the repository (optional):**
    ```
    1. get files of project
    2. cd to project loc
    ```

2.  **Install the necessary libraries using pip:**
    ```bash
    pip install opencv-python cvzone mediapipe
    ```
## How to Run

1.  **Connect a webcam:** Make sure your webcam is connected and recognized by your computer. If you have multiple cameras, you may need to change the `VideoCapture` index in the script (e.g., from `1` to `0`).

2.  **Execute the Python script:**
    ```bash
    python (one of scripts).py
    ```

3.  **To Exit:** Press the **'Esc'** key to close the application window.

## Code Overview

The script initializes the webcam and then uses `cvzone`'s `FaceDetector` and `FaceMeshDetector` to process each frame.

*   `detector.findFaces(frame)`: Locates faces and returns the frame with bounding boxes.
*   `meshdetector.findFaceMesh(frame)`: Finds facial landmarks on the faces present in the frame.

The main loop continuously reads from the camera, applies the detection and mesh, and displays the result in a window titled 'frame'.