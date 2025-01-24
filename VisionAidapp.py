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


selected_language = 'en'



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
net=cv2.dnn.readNet("//content//YOLO_README.txt")


with open("coco.names",'r') as f:
    classes=[line.strip() for line in f]

layer_names=net.getLayerNames()

output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

cap=cv2.VideoCapture(0) #you can give 1 or 2 for extra connected webcams
frame_no=0
inc=0

#Looping creates N_images to look like video
while True:
    start_time=time.time()  #starting time counting to measure frames processing speed
    _,frame=cap.read()  #reading from webcam
    frame_no+=1
    class_ids=[]
    confidences=[]
    detect_obj=0

    height,width = frame.shape[:2]  #gives dimensions of current frame

    blob=cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #detects blob(group of identicals) within the frame

    net.setInput(blob)

    outputs = net.forward(output_layers)

    for out in outputs:
        for i in out:
            scores = i[5:]                  
            class_id = np.argmax(scores)    
            confidence=scores[class_id]     
            if confidence>0.6:
                #object detected
                class_ids.append(class_id)  #all objects and their respective confidences of all blobs are stored as a list
                confidences.append(float(confidence))
engine = pyttsx3.init()

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display on the frame
    class_ids = []
    confidences = []
    boxes = []

    # Loop through detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    detected_objects = set()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)
            detected_objects.add(label)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)


    if detected_objects:
        feedback = f"I see {', '.join(detected_objects)}"
        engine.say(feedback)
        engine.runAndWait()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#
def detect_objects(frame):
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = set()
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)
            detected_objects.add(label)

    return frame, detected_objects

def speak(text):
    tts = gTTS(text)
    tts.save('output.mp3')
    display(Audio('output.mp3', autoplay=True))

from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    image = cv2.imread(filename)
    processed_image, detected_objects = detect_objects(image)

    cv2_imshow(processed_image)

    if detected_objects:
        feedback = f"I see {', '.join(detected_objects)}"
        print(feedback)
        speak(feedback)
    else:
        print("No objects detected")
        speak("No objects detected")
for text in bounds:
    detected_text = text[1]
    print(detected_text)
    engine.say(detected_text)

engine.runAndWait()

with open("img", mode='w', encoding="utf-8") as f:
    for text in bounds:
        f.write(text[1] + "\n")