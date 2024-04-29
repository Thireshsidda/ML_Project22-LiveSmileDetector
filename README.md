# ML_Project22-LiveSmileDetector


### Live Smile Detector

This project implements a real-time smile detection application using OpenCV and pre-trained cascade classifiers. It detects smiles from the webcam feed.

### Getting Started

##### Prerequisites:
Ensure you have Python 3 and OpenCV (cv2) library installed. You can install OpenCV using pip install opencv-python.
##### Download Pre-trained Classifiers:
###### Download the following pre-trained cascade classifier files:
frontal_face.xml: This file is used for face detection. You can find it online from various sources offering OpenCV pre-trained classifier data.

smile.xml: This file is specifically trained for smile detection within the detected faces. You can potentially find it alongside 

frontal_face.xml or search online for pre-trained smile cascade classifiers.

Place the downloaded XML files in the same directory as this Python script (Smile.py).


### Running the Application

Open a terminal or command prompt and navigate to the directory containing the script and classifier files.

Run the script using the following command:
```
Bash
python Smile.py
```
The application will start capturing video from your webcam and display the live feed with detected faces and smiles highlighted by rectangles.

Press Esc to quit the application.

### How it Works

The application captures video frames from your webcam.

Each frame is converted to grayscale for better processing with cascade classifiers.

The frontal_face.xml classifier is used to detect faces in the grayscale frame.

##### For each detected face:
The face region is extracted from the original frame.

The smile.xml classifier is used to detect smiles within the extracted grayscale face region.

Rectangles are drawn around detected faces (green) and smiles (blue) for visualization.

If a smile is detected, a text label "smiling" is displayed above the face.

The processed frame with detections is displayed on your screen.

### Customization

You can adjust the scaleFactor and minNeighbors parameters within the smile_detector.detectMultiScale function to fine-tune smile detection accuracy. Refer to OpenCV documentation for more details on these parameters.

Experiment with different pre-trained smile cascade classifiers (smile.xml) to see if they provide better results for your use case.


### Note

The provided smile classifier might not be perfect and may require adjustments or exploration of alternative classifiers for better accuracy.

This code is for educational purposes. Real-world smile detection applications might require more sophisticated techniques and considerations.

### Further Exploration

Explore other pre-trained cascade classifiers available for OpenCV to detect different facial features or objects.

Try building your own custom classifier for smile detection using OpenCV's machine learning capabilities.
