import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

# Fotoğraflar
f1 = cv2.imread("dusunen maymun.jpeg")
foto1 = cv2.resize(f1,(500,500))
f2 = cv2.imread("zeki maymun.jpeg")
foto2 = cv2.resize(f2,(500,500))

foto1_acik = False
foto2_acik = False

while True:
    kontrol, matris = cam.read()

    matris_rgb = cv2.cvtColor(matris, cv2.COLOR_BGR2RGB)
    results = hands.process(matris_rgb)

    index_tip_y = None 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(matris, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip_y = hand_landmarks.landmark[8].y 


  
    if index_tip_y is not None and index_tip_y > 0.5:
        if foto2_acik:
            cv2.destroyWindow("foto2")
            foto2_acik = False

        if not foto1_acik:
            cv2.imshow("foto1",foto1)
            foto1_acik = True

 
    if index_tip_y is not None and index_tip_y < 0.5:
        if foto1_acik:
            cv2.destroyWindow("foto1")
            foto1_acik = False

        if not foto2_acik:
            cv2.imshow("foto2", foto2)
            foto2_acik = True



    cv2.imshow("el deneme", matris)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
