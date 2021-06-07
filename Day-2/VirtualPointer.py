# Steps
# 1. Find hand landmarks
# 2. Get the tip of index and middle finger
# 3. 

import cv2
import numpy as np
import Resources.MPHand as mph
import time
import autopy

# predetermine variables
wc = 640
hc = 480
fr = 100 # frame rate
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0


cap = cv2.VideoCapture(0)
# setting our own width and size
# prop id of width and length is 3 and 4 resp
cap.set(3, wc)
cap.set(4, hc)
finish = False
detector = mph.HandDT()
wScr, hScr = autopy.screen.size()


while not finish:
    # 1
    s , img = cap.read()
    img = detector.find_hands(img)
    lmList, bbox = detector.find_position_pixels(img)

    # 2
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3
    fingers = detector.raise_finger_gesture()
    # print(fingers)
    cv2.rectangle(img, (fr, fr), (wc - fr, hc - fr),
    (255, 0, 255), 2)

    # 4
    if len(fingers) != 0 and fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (fr, wc - fr), (0, wScr))
        y3 = np.interp(y1, (fr, hc - fr), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
    
        # 7. Move Mouse
        autopy.mouse.move(clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 8
    if len(fingers) != 0 and fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.gaussian_distance(8, 12, img)
        print(length)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
            15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        finish = True