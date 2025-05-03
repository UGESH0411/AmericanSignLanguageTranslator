import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tkinter as tk
import pyttsx3
from PIL import Image, ImageTk
import time

model_path = "D:\\SignLanguageTranslator\\Model\\trained_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

labels = [' A  ', ' B ', ' C ', ' D ', ' E ', ' F ', ' L ']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

root = tk.Tk()
root.title("Sign Language Translator")
root.geometry("900x650")
root.configure(bg="black")

video_frame = tk.Frame(root, width=600, height=400, bg="black", relief=tk.SUNKEN, bd=3)
video_frame.pack(pady=5)

video_label = tk.Label(video_frame, bg="black")
video_label.pack()

detected_label = tk.Label(root, text="Detected Letter: ", font=("Arial", 18), fg="yellow", bg="black")
detected_label.pack(pady=5)

word_label = tk.Label(root, text="Word Forming: ", font=("Arial", 18), fg="#00FFAA", bg="black")
word_label.pack(pady=5)

button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=10)

reset_button = tk.Button(button_frame, text="Reset", font=("Arial", 14), bg="red", fg="white", command=lambda: reset_text())
reset_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", font=("Arial", 14), bg="orange", fg="black", command=lambda: clear_letter())
clear_button.grid(row=0, column=1, padx=10)

speech_button = tk.Button(button_frame, text="Speech", font=("Arial", 14), bg="green", fg="white", command=lambda: speak_word())
speech_button.grid(row=0, column=2, padx=10)

exit_button = tk.Button(button_frame, text="Exit", font=("Arial", 14), bg="darkgray", fg="black", command=root.quit)
exit_button.grid(row=0, column=3, padx=10)

word_sequence = []
previous_letter = None
stable_threshold = 5
letter_counter = {}
last_added_time = time.time()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def reset_text():
    global word_sequence, letter_counter, previous_letter
    word_sequence = []
    letter_counter = {}
    previous_letter = None
    word_label.config(text="Word Forming: ", fg="#00FFAA")
    detected_label.config(text="Detected Letter: ", fg="yellow")

def clear_letter():
    detected_label.config(text="Detected Letter: ", fg="yellow")

def speak_word():
    engine = pyttsx3.init()
    engine.say("".join(word_sequence))
    engine.runAndWait()

def process_video():
    global previous_letter, letter_counter, last_added_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, process_video)
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        nearest_hand = None
        max_box_size = 0

        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            box_size = (x_max - x_min) * (y_max - y_min)
            if box_size > max_box_size:
                max_box_size = box_size
                nearest_hand = hand_landmarks

        if nearest_hand:
            x_min = max(0, int(min([lm.x for lm in nearest_hand.landmark]) * w))
            y_min = max(0, int(min([lm.y for lm in nearest_hand.landmark]) * h))
            x_max = min(w, int(max([lm.x for lm in nearest_hand.landmark]) * w))
            y_max = min(h, int(max([lm.y for lm in nearest_hand.landmark]) * h))

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                root.after(10, process_video)
                return

            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = np.expand_dims(hand_img, axis=0) / 255.0

            prediction = model.predict(hand_img)
            confidence = np.max(prediction)
            label = labels[np.argmax(prediction)]

            current_time = time.time()

            if confidence >= 0.57:
                letter_counter[label] = letter_counter.get(label, 0) + 1

                if letter_counter[label] >= stable_threshold:
                    if label != previous_letter or (current_time - last_added_time > 1.0):
                        detected_label.config(text=f"Detected Letter: {label}", fg="yellow")
                        word_sequence.append(label)
                        word_label.config(text=f"Word Forming: {''.join(word_sequence)}", fg="#00FFAA")
                        previous_letter = label
                        last_added_time = current_time

                    letter_counter = {label: stable_threshold}

            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, nearest_hand, mp_hands.HAND_CONNECTIONS)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((600, 400))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(10, process_video)

root.after(10, process_video)
root.mainloop()
cap.release()
cv2.destroyAllWindows()
