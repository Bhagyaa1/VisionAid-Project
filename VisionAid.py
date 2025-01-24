import cv2
import numpy as np
from gtts import gTTS
import threading
import os
import time
from datetime import datetime  # Correct import
from queue import Queue
from flask import Flask, render_template, redirect, url_for
from deep_translator import GoogleTranslator
import pygame  # For audio playback

app = Flask(__name__)

# Global variable for language selection
selected_language = 'en'

# Define messages for each object with navigation instructions
object_messages = {
    'person': "A person is nearby. Please stop and give way.",
    'bicycle': "A bicycle is approaching on your left. Be careful.",
    'car': "A car is nearby, coming from the right. Please move to safety.",
    'motorbike': "Watch out! A motorbike is approaching from behind.",
    'aeroplane': "An aeroplane is overhead. No immediate action needed.",
    'bus': "Attention! A bus is approaching. Move away from the curb.",
    'train': "Warning! A train is coming. Please stay clear of the tracks.",
    'truck': "Caution! A truck is approaching. Keep clear of its path.",
    'boat': "A boat is nearby on the water. No action needed.",
    'traffic light': "A traffic light is ahead. Follow the signals for safety.",
    'fire hydrant': "Fire hydrant detected. Please do not block access.",
    'stop sign': "Stop sign detected. Prepare to stop.",
    'parking meter': "Parking meter detected. Stay clear.",
    'bench': "A bench is nearby. You can rest here if needed.",
    'bird': "A bird is nearby. No immediate action needed.",
    'cat': "A cat is nearby. Approach carefully.",
    'dog': "A dog is approaching. Be friendly but cautious.",
    'horse': "A horse is near. Keep a safe distance.",
    'sheep': "Sheep detected. Be mindful while walking.",
    'cow': "A cow is nearby. Keep your distance for safety.",
    'elephant': "An elephant is nearby. Maintain a safe distance.",
    'bear': "Warning! A bear is detected. Move away immediately!",
    'zebra': "A zebra is nearby. Observe from a distance.",
    'giraffe': "A giraffe is in the area. Observe quietly.",
    'backpack': "A backpack is nearby. Ensure it's secure.",
    'umbrella': "Umbrella detected. Be aware of the weather.",
    'handbag': "Handbag detected. Keep it close.",
    'tie': "Tie detected. No action needed.",
    'suitcase': "Suitcase detected. Be cautious of your belongings.",
    'frisbee': "Frisbee detected. Watch out for flying objects!",
    'skis': "Skis detected. Be cautious in snowy areas.",
    'snowboard': "Snowboard detected. Watch out for snowy paths.",
    'sports ball': "A sports ball is nearby. Avoid the playing area.",
    'kite': "A kite is flying nearby. Stay clear of the area.",
    'baseball bat': "Baseball bat detected. Watch for active play.",
    'baseball glove': "Baseball glove detected. Be aware of nearby activities.",
    'skateboard': "Skateboard detected. Stay clear of skaters.",
    'surfboard': "Surfboard detected. Keep away from water activities.",
    'tennis racket': "Tennis racket detected. Stay clear of courts.",
    'bottle': "Bottle detected. Ensure it's not left on the ground.",
    'wine glass': "Wine glass detected. Be careful around fragile items.",
    'cup': "Cup detected. Watch your step.",
    'fork': "Fork detected. Handle carefully.",
    'knife': "Knife detected. Stay cautious.",
    'spoon': "Spoon detected. No immediate action needed.",
    'bowl': "Bowl detected. Be careful around food areas.",
    'banana': "Banana detected. Watch for slipping hazards.",
    'apple': "Apple detected. No immediate action needed.",
    'sandwich': "Sandwich detected. Be careful around food.",
    'orange': "Orange detected. No immediate action needed.",
    'broccoli': "Broccoli detected. No immediate action needed.",
    'carrot': "Carrot detected. No immediate action needed.",
    'hot dog': "Hot dog detected. Watch for food items.",
    'pizza': "Pizza detected. Be cautious around food.",
    'donut': "Donut detected. Watch for food items.",
    'cake': "Cake detected. Be aware of food items.",
    'chair': "Chair detected. Be careful when moving.",
    'sofa': "Sofa detected. Be cautious when sitting.",
    'pottedplant': "Potted plant detected. Avoid bumping into it.",
    'bed': "Bed detected. Be cautious around sleeping areas.",
    'diningtable': "Dining table detected. Watch your step.",
    'toilet': "Toilet detected. Please use when needed.",
    'tvmonitor': "TV monitor detected. No action needed.",
    'laptop': "Laptop detected. Be cautious of your belongings.",
    'mouse': "Mouse detected. No immediate action needed.",
    'remote': "Remote detected. No action needed.",
    'keyboard': "Keyboard detected. No action needed.",
    'cell phone': "Cell phone detected. Keep it secure.",
    'microwave': "Microwave detected. Be cautious in kitchens.",
    'oven': "Oven detected. Watch for hot surfaces.",
    'toaster': "Toaster detected. Watch for hot surfaces.",
    'sink': "Sink detected. Be cautious of water hazards.",
    'refrigerator': "Refrigerator detected. Keep it closed.",
    'book': "Book detected. Be careful not to trip.",
    'clock': "Clock detected. No action needed.",
    'vase': "Vase detected. Watch out for fragile items.",
    'scissors': "Scissors detected. Handle with care.",
    'teddy bear': "Teddy bear detected. A soft friend nearby.",
    'hair drier': "Hair dryer detected. Be cautious of hot surfaces.",
    'toothbrush': "Toothbrush detected. No action needed."
}


# Load YOLO weights and configurations
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Define classes for detection
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

run_detection = False
audio_lock = threading.Lock()  # Mutex for audio playback
audio_queue = Queue()  # Queue for audio playback


def play_audio():
    pygame.mixer.init()  # Initialize pygame mixer
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:
            break
        with audio_lock:
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():  # Wait for the music to finish playing
                    time.sleep(0.1)  # Avoid busy waiting
            except Exception as e:
                print(f"Error playing audio: {e}")

def audio_thread():
    play_audio()

threading.Thread(target=audio_thread, daemon=True).start()

def detect_objects():
    global run_detection, selected_language
    cap = cv2.VideoCapture(0)
    
    while run_detection:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[layer_id - 1] for layer_id in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                detected_class = classes[class_ids[i]]
                object_name = object_messages.get(detected_class, f"{detected_class} detected. Please proceed with caution.")

                try:
                    # translated_message = GoogleTranslator().translate(object_name, target_language=selected_language)
                    translated_message = GoogleTranslator(source='en', target=selected_language).translate(object_name)
                    print(f"Translated message: {translated_message} in language: {selected_language}")  # Debug line
                except Exception as e:
                    print(f"Translation error: {e}")
                    translated_message = object_name  # Fallback to original message


                try:
                    print(f"Generating TTS for message: {translated_message} in language: {selected_language}")  # Debug line
                    tts = gTTS(text=translated_message, lang=selected_language)
                    audio_file = f"temp/temp_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
                    tts.save(audio_file)
                    audio_queue.put(audio_file)
                except Exception as e:
                    print(f"TTS error: {e}")


        time.sleep(10)

    cap.release()


@app.route('/set_language/<lang>')
def set_language(lang):
    global selected_language
    selected_language = lang
    print(f"Language set to: {selected_language}")  # Debug line
    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection')
def start_detection():
    global run_detection
    if not run_detection:
        run_detection = True
        threading.Thread(target=detect_objects).start()
    return redirect(url_for('index'))

@app.route('/stop_detection')
def stop_detection():
    global run_detection
    run_detection = False
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

