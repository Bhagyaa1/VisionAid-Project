# VisionAid-Project

A real-time assistive solution for visually impaired individuals, empowering them to navigate their surroundings and access information with ease.

Key Features
Image Processing: Captures and enhances image quality for improved analysis.
Text-to-Speech (TTS): Converts extracted text to natural-sounding speech using Google TTS and pyttsx3.
OCR Integration: Utilizes the Google Vision API to detect and extract text from images with high accuracy.
Object Recognition: Employs YOLO and a custom dataset tailored for the Indian environment (e.g., identifying doors, lockers, stairs).
Voice Command Interaction: Users can control the app through voice commands like "Open Camera" and "Start Object Recognition."
Navigation Support: Provides real-time navigation guidance for safe and easy mobility.
Custom Dataset 📚
Integrated the MS COCO 2017 dataset with 80 classes and enriched it with 15 additional classes relevant to Indian settings (e.g., manholes, tree, postbox).
A dataset of 5000+ images with consistent annotations.
Technologies Used 🛠️
Languages: Python
APIs: Google Vision API, Text-to-Speech APIs
Deep Learning: YOLO for object detection
Frameworks: TensorFlow, Keras
Development: Google Colab
