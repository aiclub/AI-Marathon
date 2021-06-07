'''
This is a code along session
'''
import cv2

# predetermine variables
wc = 640
wh = 480

cap = cv2.VideoCapture(0)
# setting our own width and size
# prop id of width and length is 3 and 4 resp
cap.set(3, wc)
cap.set(4, wh)
finish = False


while not finish:
    s , img = cap.read()
    cv2.imshow("Image", img)

    
    if cv2.waitKey(1) == 27:
        finish = True