import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Set up camera
wCam, hCam = 640,480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.5)





devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]
vol=0
volBAR=400
Percentage=0

while True:
    success, img = cap.read()

    # Find hands and get landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)  # Pass draw=False to avoid extra circles



    if len(lmList) != 0:
        # Get coordinates for the thumb tip and index finger tip
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
        cx,cy=(x1+x2)//2 ,(y1+y2)//2

        # Draw circles on the thumb and index finger tips
        cv2.circle(img, (x1, y1), 11, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 11, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255,0), 3)
        cv2.circle(img, (cx, cy), 11, (255, 0, 255), cv2.FILLED)

        length=math.hypot(x2-x1,y2-y1)
       # print(length)

        #hand range 50-250
        #volume range -45 to 0
        vol=np.interp(length,[50,200],[minVol,maxVol])
        Percentage = np.interp(length, [50, 200], [0, 100])
        volBAR = np.interp(length, [50, 200], [400, 150])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)





        if length<50:
            cv2.circle(img, (cx, cy), 11, (128, 0, 0), cv2.FILLED)
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBAR)),(85,400),(0,255,0),cv2.FILLED)#fills the rect as per the volume
    cv2.putText(img, f'{int(Percentage)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (259, 0, 51), 3)

    # Print the landmarks to verify
       # print("Thumb tip:", lmList[4], "Index tip:", lmList[8])

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on image with navy blue color
    cv2.putText(img, f'FPS:{int(fps)}', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show image
    cv2.imshow("Img", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
