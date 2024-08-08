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




SMOOTHING=7
screenwidth , screenheight = pyautogui.size()
prevx = 0
prevy = 0
# set up sensitivity
sensitivity = 9

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

                     # Calculate the midpoint between thumb tip and index finger tip
                thumb_tip = hand.landmark[THUMB_TIP_IDX]
                index_tip = hand.landmark[INDEX_FINGER_TIP_IDX]
                index_mcp = hand.landmark[INDEX_FINGER_MCP_IDX]

                midpoint_x = (thumb_tip.x + index_tip.x) / 2
                midpoint_y = (thumb_tip.y + index_tip.y) / 2

                # Map midpoint to screen coordinates
                screen_x = np.interp(midpoint_x, [0, 1], [0, screenwidth *sensitivity])
                screen_y = np.interp(midpoint_y, [0, 1], [0, screenheight * sensitivity])

                # Smooth cursor movement
                current_x = prevx + (screen_x - prevx) / SMOOTHING
                current_y = prevy + (screen_y - prevy) / SMOOTHING

                # Move cursor
                pyautogui.moveTo(screenwidth - current_x, current_y)
                prev_x, prev_y = current_x, current_y


                if (thumb_tip.y < index_mcp.y):
                    pyautogui.leftClick()


    cv2.imshow("CamOutput",img)
    cv2.waitKey(1)

