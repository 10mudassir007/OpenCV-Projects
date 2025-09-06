import cv2
import mediapipe as mp
import pyautogui

# Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

screen_w, screen_h = pyautogui.size()  # get screen size
cam_w, cam_h = 640, 480  # camera resolution

cap = cv2.VideoCapture(0)  # replace 0 with your Iriun webcam index
cap.set(3, cam_w)
cap.set(4, cam_h)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Index fingertip
        x_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        y_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

        # Convert to screen coordinates
        x_screen = int(x_tip * screen_w)
        y_screen = int(y_tip * screen_h)

        # Move mouse
        pyautogui.moveTo(x_screen, y_screen)

        # Optional: detect click (thumb + index finger distance)
        x_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        y_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        distance = ((x_tip - x_thumb)**2 + (y_tip - y_thumb)**2)**0.5

        if distance < 0.05:  # adjust threshold
            pyautogui.click()

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()