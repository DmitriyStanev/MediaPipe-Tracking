import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__ (self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks,
                                           self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmarkListXY  = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                # if id == 4:
                #     print("ID ---> ", id, "Landmark --->", landmark)

                x, y = landmark.x*100, landmark.y*100
                z = landmark.z*100

                landmarkListXY.append([id, int(x), int(y), int(z)])

        return landmarkListXY

    def checkFingers(self, img, landmarkListXY):
        thumbIsOpen = False
        indexFingerIsOpen = False
        middleFingerIsOpen = False
        ringFingerIsOpen = False
        pinkyFingerIsOpen = False

        if len(landmarkListXY) != 0:
            if landmarkListXY[3][2] < landmarkListXY[2][2] and landmarkListXY[4][2] < landmarkListXY[2][2]:
                thumbIsOpen = True
            else:
                thumbIsOpen = False

            if landmarkListXY[7][2] < landmarkListXY[6][2] and landmarkListXY[8][2] < landmarkListXY[6][2]:
                indexFingerIsOpen = True
            else:
                indexFingerIsOpen = False

            if landmarkListXY[11][2] < landmarkListXY[10][2] and landmarkListXY[12][2] < landmarkListXY[10][2]:
                middleFingerIsOpen = True
            else:
                middleFingerIsOpen = False

            if landmarkListXY[15][2] < landmarkListXY[14][2] and landmarkListXY[16][2] < landmarkListXY[14][2]:
                ringFingerIsOpen = True
            else:
                ringFingerIsOpen = False

            if landmarkListXY[19][2] < landmarkListXY[18][2] and landmarkListXY[20][2] < landmarkListXY[18][2]:
                pinkyFingerIsOpen = True
            else:
                pinkyFingerIsOpen = False

        fingers = thumbIsOpen, indexFingerIsOpen, middleFingerIsOpen, ringFingerIsOpen, pinkyFingerIsOpen

        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            cv2.putText(img, "Peace!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
            cv2.putText(img, "Spider Man!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            cv2.putText(img, "One!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if not fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            cv2.putText(img, "OKAY!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            cv2.putText(img, "Four!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        # if not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        #     cv2.putText(img, "Like!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
        #                 (0, 0, 255), 4)

        if fingers[0] and not fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            cv2.putText(img, "Not good!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        # return thumbIsOpen, firstFingerIsOpen, middleFingerIsOpen, thirdFingerIsOpen, fourthFingerIsOpen




def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        # landmarkList = detector.findPosition(img)
        landmarkListXY = detector.findPosition(img)
        # if len(landmarkListXY) != 0:
        #     print(landmarkListXY[0], landmarkListXY[1], landmarkListXY[2], landmarkListXY[3], landmarkListXY[4])

        detector.checkFingers(img, landmarkListXY)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 4)

        cv2.imshow("MediaPipe Hands", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()