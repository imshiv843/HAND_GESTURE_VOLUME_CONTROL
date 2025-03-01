import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# DECLARING THE FRAME RATE
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                print(id, cx, cy)

                # ✅ Fixed Indentation Here
                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (225, 105, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS Calculation
    currentTime = time.time()
    fps = 1 / (currentTime - pTime)
    pTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 4)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
