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

                x, y = landmark.x*100, landmark.y*100
                z = landmark.z*100

                landmarkListXY.append([id, int(x), int(y), int(z)])

        return landmarkListXY

    def checkFingers(self, img, landmarkListXY):

        fingers = {'thumbIsOpen': False, 'indexFingerIsOpen': False, 'middleFingerIsOpen': False,
                   'ringFingerIsOpen': False, 'pinkyFingerIsOpen': False, 'crossHands': True}

        if len(landmarkListXY) != 0:

            if abs(landmarkListXY[4][1] - landmarkListXY[9][1]) > 10 or \
                    abs(landmarkListXY[4][2] - landmarkListXY[9][2]) > 10:
                fingers['thumbIsOpen'] = True

            if landmarkListXY[7][2] < landmarkListXY[6][2] and landmarkListXY[8][2] < landmarkListXY[6][2]:
                fingers['indexFingerIsOpen'] = True

            if landmarkListXY[11][2] < landmarkListXY[10][2] and landmarkListXY[12][2] < landmarkListXY[10][2]:
                fingers['middleFingerIsOpen'] = True

            if landmarkListXY[15][2] < landmarkListXY[14][2] and landmarkListXY[16][2] < landmarkListXY[14][2]:
                fingers['ringFingerIsOpen'] = True

            if landmarkListXY[19][2] < landmarkListXY[18][2] and landmarkListXY[20][2] < landmarkListXY[18][2]:
                fingers['pinkyFingerIsOpen'] = True

        cv2.putText(img, ('Thumb Finger is open: ' + str(fingers['thumbIsOpen'])), (10, 180),
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 200), 2)
        cv2.putText(img, ('Index Finger is open: ' + str(fingers['indexFingerIsOpen'])), (10, 200),
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 200), 2)
        cv2.putText(img, ('Middle Finger is open: ' + str(fingers['middleFingerIsOpen'])), (10, 220),
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 200), 2)
        cv2.putText(img, ('Ring Finger is open: ' + str(fingers['ringFingerIsOpen'])), (10, 240),
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 200), 2)
        cv2.putText(img, ('Pinky Finger is open: ' + str(fingers['pinkyFingerIsOpen'])), (10, 260),
                    cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 200), 2)

        if fingers['indexFingerIsOpen'] and fingers['middleFingerIsOpen'] and \
                not fingers['ringFingerIsOpen'] and not fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "Peace!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if fingers['thumbIsOpen'] and fingers['indexFingerIsOpen'] and not fingers['middleFingerIsOpen'] and\
                not fingers['ringFingerIsOpen'] and fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "Spider Man!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if not fingers['thumbIsOpen'] and fingers['indexFingerIsOpen'] and not fingers['middleFingerIsOpen'] and \
                not fingers['ringFingerIsOpen'] and fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "Let's ROCK!!!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if not fingers['thumbIsOpen'] and fingers['indexFingerIsOpen'] and not fingers['middleFingerIsOpen'] and \
                not fingers['ringFingerIsOpen'] and not fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "One!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if fingers['thumbIsOpen'] and not fingers['indexFingerIsOpen'] and fingers['middleFingerIsOpen'] and \
                fingers['ringFingerIsOpen'] and fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "OKAY!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if not fingers['thumbIsOpen'] and fingers['indexFingerIsOpen'] and fingers['middleFingerIsOpen'] and \
                fingers['ringFingerIsOpen'] and fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "Four!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if fingers['thumbIsOpen'] and not fingers['indexFingerIsOpen'] and not fingers['middleFingerIsOpen'] and \
                not fingers['ringFingerIsOpen'] and not fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "Like!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        if not fingers['thumbIsOpen'] and not fingers['indexFingerIsOpen'] and fingers['middleFingerIsOpen'] and \
                not fingers['ringFingerIsOpen'] and not fingers['pinkyFingerIsOpen']:
            cv2.putText(img, "Not good!", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 4)

        return fingers


def main():
    previousTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarkListXY = detector.findPosition(img)

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