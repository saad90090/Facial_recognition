# Facial Recognition Attendance System using FaceNet

## Project Overview  
This project implements a facial recognition attendance system using FaceNet embeddings. It uses cosine similarity to match faces captured via webcam and records attendance accordingly. The system is built with Python and Flask and can use your phone as a webcam via DroidCam.

## Features  
- Real-time facial recognition for attendance  
- Uses FaceNet model embeddings for accurate face matching  
- Attendance marking based on cosine similarity of embeddings  
- Web-based interface powered by Flask  
- Support for phone webcam input using DroidCam  

## Technologies Used  
- Python  
- Flask (Web framework)  
- OpenCV (Webcam and image processing)  
- DeepFace (Facial recognition model wrapper)  
- scikit-learn (Normalizer and cosine similarity)  
- DroidCam (for using phone as webcam)  
- Werkzeug (for secure file handling)  

## Installation and Setup

pip install flask opencv-python numpy deepface scikit-learn werkzeug
Note: DroidCam app needs to be installed separately on your phone and PC to use the phone as a webcam.
python app.py
Open your browser at http://localhost:5000 to access the attendance system.

Usage
Connect your phone via DroidCam or use your PC webcam.

The system will capture your face, compute embeddings, and check attendance by comparing with stored embeddings using cosine similarity.

Attendance is recorded with timestamp.

File Structure
app.py: Main Flask application

templates/: HTML templates for the web interface

attendance.csv: CSV file storing attendance records

registered_embeddings.npy: Stores the embeddings for the images
registered_labels.json: Stores the names for the people added

Acknowledgments
FaceNet model via DeepFace library

OpenCV for real-time image processing

DroidCam for mobile webcam support
