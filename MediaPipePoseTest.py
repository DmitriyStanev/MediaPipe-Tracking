import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(upper_body_only=False)
mp_draw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## Pose object uses only RGB images
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            # print(id, landmark)
            height, width, channel = img.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            print(id, cx, cy)
            if id == 0:
                cv2.circle(img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("MediaPipe Pose", img)
    cv2.waitKey(1)