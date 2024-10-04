import cv2
import mediapipe as mp

mp_maos = mp.solutions.hands
maos = mp_maos.Hands()
mp_desenho = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado = maos.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)

    cv2.imshow('Detecção de Mãos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
