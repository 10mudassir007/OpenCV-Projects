import cv2
import mediapipe as mp
import numpy as np

# EAR threshold — below this is "closed"
EAR_THRESHOLD = 0.25

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye indices based on Mediapipe FaceMesh model
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye_landmarks):
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    status = "NO FACE"
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [
                [
                    int(face_landmarks.landmark[i].x * w),
                    int(face_landmarks.landmark[i].y * h),
                ]
                for i in LEFT_EYE
            ]
            right_eye = [
                [
                    int(face_landmarks.landmark[i].x * w),
                    int(face_landmarks.landmark[i].y * h),
                ]
                for i in RIGHT_EYE
            ]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0
            status = "Awake" if avg_ear > EAR_THRESHOLD else "Sleeping"

            cv2.putText(
                image,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0) if status == "Awake" else (0, 0, 255),
                3,
            )

    cv2.imshow("Eye State Detector", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
