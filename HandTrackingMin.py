import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandmarks.landmark):
                print("ID ---> ", id,"Landmark --->", landmark)
                hight, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * hight)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 4)

    cv2.imshow("MediaPipe Hands", img)
    cv2.waitKey(1)