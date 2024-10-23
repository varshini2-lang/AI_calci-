import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, maxHands=1, staticMode=False, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.maxHands = maxHands
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=staticMode,
                                        max_num_hands=maxHands,
                                        min_detection_confidence=detectionCon,
                                        min_tracking_confidence=minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, flipType=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        hands = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                hands.append(handLms)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img, hands

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
        return lmList

    def fingersUp(self, handLms):
        fingers = [0, 0, 0, 0, 0]  # Assume 5 fingers are down initially

        if handLms:
            # Example logic for detecting fingers up based on landmarks
            # Check if fingers are up by comparing landmarks for each finger
            if handLms.landmark[8].y < handLms.landmark[6].y:  # Index finger
                fingers[1] = 1
            if handLms.landmark[12].y < handLms.landmark[10].y:  # Middle finger
                fingers[2] = 1
            if handLms.landmark[16].y < handLms.landmark[14].y:  # Ring finger
                fingers[3] = 1
            if handLms.landmark[20].y < handLms.landmark[18].y:  # Pinky
                fingers[4] = 1
            if handLms.landmark[4].x < handLms.landmark[2].x:  # Thumb (x comparison because thumb moves sideways)
                fingers[0] = 1

        return fingers
