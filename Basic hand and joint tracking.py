import cv2
import mediapipe as mp
import time
from subprocess import call
import numpy as np
import pyautogui
WRIST_IDX = 0
THUMB_CNC_IDX=1
THUMB_MCP_IDX=2
THUMB_IP_IDX=3
THUMB_TIP_IDX=4
INDEX_FINGER_MCP_IDX=5
INDEX_FINGER_PIP_IDX=6
INDEX_FINGER_DIP_IDX=7
INDEX_FINGER_TIP_IDX=8
MIDDLE_FINGER_MCP_IDX=9
MIDDLE_FINGER_PIP_IDX=10
MIDDLE_FINGER_DIP_IDX=11
MIDDLE_FINGER_TIP_IDX=12
RING_FINGER_MCP_IDX=13
RING_FINGER_PIP_IDX=14
RING_FINGER_DIP_IDX=15
RING_FINGER_TIP_IDX=16
PINKY_MCP_IDX=17
PINKY_PIP_IDX=18
PINKY_DIP_IDX=19
PINKY_TIP_IDX=20




Smoothen=7

videoCap=cv2.VideoCapture(0)
lastFrameTime = 0
frame = 0
max_diff=0
min_diff = 100000
handSolution = mp.solutions.hands
hands = handSolution.Hands()

while True:
    frame +=1
    success,img = videoCap.read()
    if success :
        #reading current image
        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # checks the frames per second
        thisFrameTime=time.time()
        fps=1/(thisFrameTime-lastFrameTime)
        lastFrameTime=thisFrameTime
        #make fps appear on the image 
        cv2.putText(img,F'FPS:{int(fps)}',
                    (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #recognise hands
        recHands= hands.process(img)
        if recHands.multi_hand_landmarks:
            for hand in recHands.multi_hand_landmarks:
            #apply dots to hand joints
                for datapoint_id,point in enumerate(hand.landmark):
                    h,w,c = img.shape
                    x,y = int(point.x * w),int(point.y * h)
                    cv2.circle(img,(x,y),
                               10,(255,255,255)
                               ,cv2.FILLED)
    cv2.imshow("CamOutput",img)
    cv2.waitKey(1)



