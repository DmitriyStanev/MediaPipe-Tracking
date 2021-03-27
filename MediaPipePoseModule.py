import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, upperBodyOnly=False, smoothLandmarks=True,
                 detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.upperBodyOnly = upperBodyOnly
        self.smoothLandmarks = smoothLandmarks
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBodyOnly, self.smoothLandmarks,
                                     self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  ## Pose object uses only RGB images
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        landmarkList = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                # print(id, landmark)
                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)
                landmarkList.append([id, cx, cy])
                if id == 0:
                    cv2.circle(img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)
        return landmarkList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("MediaPipe Pose", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()