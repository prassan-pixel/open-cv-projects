import numpy as np
import cv2 as cv
import mediapipe as mp
from collections import deque

def setValues(x):
    print("")

cv.namedWindow("Color detectors")

cv.createTrackbar("upper hue","Color detectors",153,180,setValues)
cv.createTrackbar("upper saturation","Color detectors",255,255,setValues)
cv.createTrackbar("upper value","Color detectors",255,255,setValues)

cv.createTrackbar("lower hue","Color detectors",64,180,setValues)
cv.createTrackbar("lower saturation","Color detectors",72,255,setValues)
cv.createTrackbar("lower value","Color detectors",49,255,setValues)

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5,5),np.uint8)


colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorIndex = 0

paintWindow = np.zeros((471,636,3),dtype= np.uint8) + 255

paintWindow = cv.rectangle(paintWindow,(40,1),(140,65),(0,0,0),2)
paintWindow = cv.rectangle(paintWindow,(160,1),(255,65),colors[0],-1)
paintWindow = cv.rectangle(paintWindow,(275,1),(370,65),colors[1],-1)
paintWindow = cv.rectangle(paintWindow,(390,1),(485,65),colors[2],-1)
paintWindow = cv.rectangle(paintWindow,(505,1),(600,65),colors[3],-1)

cv.putText(paintWindow,"CLEAR",(49,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
cv.putText(paintWindow,"BLUE",(199,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
cv.putText(paintWindow,"GREEN",(300,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
cv.putText(paintWindow,"RED",(420,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
cv.putText(paintWindow,"YELLOW",(520,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)


cap = cv.VideoCapture(0)
while True:
    Success,frame = cap.read()
    frame = cv.flip(frame,1)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    u_hue = cv.getTrackbarPos("upper hue","Color detectors")
    u_saturation = cv.getTrackbarPos("upper saturation","Color detectors")
    u_value = cv.getTrackbarPos("upper value","Color detectors")

    l_hue = cv.getTrackbarPos("lower hue","Color detectors")
    l_saturation = cv.getTrackbarPos("lower saturation","Color detectors")
    l_value = cv.getTrackbarPos("lower value","Color detectors")

    upper_hsv = np.array([u_hue,u_saturation,u_value])
    lower_hsv = np.array([l_hue,l_saturation,l_value])

    frame = cv.rectangle(frame,(40,1),(140,65),(0,0,0),-1)
    frame = cv.rectangle(frame,(160,1),(255,65),colors[0],-1)
    frame = cv.rectangle(frame,(275,1),(370,65),colors[1],-1)
    frame = cv.rectangle(frame,(390,1),(485,65),colors[2],-1)
    frame = cv.rectangle(frame,(505,1),(600,65),colors[3],-1)

    cv.putText(frame,"CLEAR",(49,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame,"BLUE",(199,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
    cv.putText(frame,"GREEN",(300,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
    cv.putText(frame,"RED",(420,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
    cv.putText(frame,"YELLOW",(520,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)

    mask = cv.inRange(hsv,lower_hsv,upper_hsv)
    mask = cv.erode(mask,kernel,iterations=1)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel) #removing small noises
    mask = cv.dilate(mask,kernel,iterations=1)
    

    cnts,z = cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv.contourArea, reverse=True)[0]
        ((x,y),radius) = cv.minEnclosingCircle(cnt)
        cv.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        M = cv.moments(cnt)
        # M['m10'] and M['m01']: First-order moments
        # M['m00'] : Zeroth-order moments (area of the contour)
        center = (int(M['m10'] / M['m00']),int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 40 <= center[0] <= 140:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0 #blue
            elif 275 <= center[0] < 370:
                colorIndex = 1 #green
            elif 390 <= center[0] < 485:
                colorIndex = 2 #red
            elif 505 <= center[0] < 600:
                colorIndex = 3 #yellow
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bpoints,gpoints,rpoints,ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv.line(paintWindow,points[i][j][k-1],points[i][j][k],colors[i],2)
    

    cv.imshow('Live Drawing',frame)
    cv.imshow('white canvas',paintWindow)
    cv.imshow('mask',mask)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()