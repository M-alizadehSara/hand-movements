import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import pyplot as plt
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
prev_action = None
action = None
images = os.listdir("images")
currentIndex = 0
lastIndex = 0


def detect_gesture(landmarks):
    global prev_action, action, currentIndex
    x_tip = landmarks[8].x
    x_base = landmarks[5].x
    if abs(x_tip - x_base) > 0.1:
        if x_tip > x_base + 0.1:
            action = 1
        elif x_tip < x_base - 0.1:
            action = 0
        else:
            action = None
        if action is not None:
            print(f"Action: {action}")
            prev_action = action


def showImage():
    global currentIndex
    global lastIndex
    global action
    fig, ax = plt.subplots()
    img = mpimg.imread(f'./images/{images[0]}')
    ax.imshow(img)
    ax.axis('off')
    plt.draw()
    lastIndex = currentIndex
    while True:
        if action == 1:
            currentIndex += 1
        elif action == 0:
            currentIndex -= 1
        if currentIndex > len(images) - 1:
            currentIndex = 0
        elif currentIndex < 0:
            currentIndex = len(images) - 1
        if currentIndex != lastIndex:
            img = mpimg.imread(f'./images/{images[currentIndex]}')
            ax.imshow(img)
            ax.axis('off')
            plt.draw()
            lastIndex = currentIndex
            action = None
        if action == -1:
            return None
        plt.pause(0.3)


def main():
    global action
    cap = cv2.VideoCapture(0)
    threading.Thread(target=showImage).start()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                detect_gesture(hand_landmarks.landmark)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Movement Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            action = -1
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
