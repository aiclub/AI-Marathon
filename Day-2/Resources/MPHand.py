import cv2
import mediapipe as mp
import time
import math

class HandDT():
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Setting up media pipe hand tracker
        self.mpHands = mp.solutions.hands
        # by default static_image_mode is set to false so that detection occurs only when tracking
        # fall below a certain threshhold
        self.hands = self.mpHands.Hands(static_image_mode,
                                        max_num_hands,
                                        min_detection_confidence,
                                        min_tracking_confidence)
        # To draw lines between 21 points
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(
                        img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def raise_finger_gesture(self):
        fingers = []
        if len(self.lm_list) != 0:
            # thumb
            if self.lm_list[4][1] > self.lm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(2,6):
                if self.lm_list[id*4][2] <  self.lm_list[id*4-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def gaussian_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        distance = math.hypot(x2-x1, y2-y1)
        mx, my = (x1+x2)//2, (y1+y2)//2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (mx, my), r, (255,0,255), cv2.FILLED)

        return distance, img, [x1, y1, x2, y2, mx, my]


    def find_position_pixels(self, img, hand_no=0, draw=False):
        xList = []
        yList = []
        bbox = []
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lm_list.append([id, cx, cy])
                xList.append(cx)
                yList.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        if len(xList) != 0 and len(yList) != 0:
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                (0, 255, 0), 2)
        return self.lm_list, bbox

    def get_index_finger_tip(self, img, hand_no=0, draw=False):
        lm = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            lm = myHand.landmark[8]
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if draw:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lm


def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDT()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position_pixels(img)
        ind_fi = detector.get_index_finger_tip(img, draw=True)
        # if len(lm_list) != 0:
        #     print(lm_list[4])

        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Images", img)
        if cv2.waitKey(1) == 27:
            finish = True


if __name__ == "__main__":
    main()