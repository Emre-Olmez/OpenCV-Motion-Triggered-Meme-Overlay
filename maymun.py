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
    kontrol, matris = cam.read()#MATRİSİ OKUTMAMIZ LAZIM

    matris_rgb = cv2.cvtColor(matris, cv2.COLOR_BGR2RGB)#PARMAK OKUYUCU RGB OKUSUN DİYE RGB DÖNDÜRDÜK
    results = hands.process(matris_rgb)#OKUTMA İŞLEMİNİ RGB ÜZERİNDEN YAPTIK VE RESULTS DEĞİŞKENİNE ALDIK

    index_tip_y = None   # İşaret parmağı

    if results.multi_hand_landmarks:#BİRDEN FAZLA ELİ ALGILAMASI İÇİN İŞLEM
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(matris, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip_y = hand_landmarks.landmark[8].y     # İşaret parmağı ucu İNDEXİ HEP 8 DİR KÜTÜPHANEYE ÖZEL SONDAKİ Y KOORDİNAT


    #İŞARET PARMAĞI EKRANA GÖRE 0.5 DEN FAZLAYSA FOTO2 AÇ
    if index_tip_y is not None and index_tip_y > 0.5:
        if foto2_acik:
            cv2.destroyWindow("foto2")#FOTO2 AÇIKSA KAPAT EVLAT
            foto2_acik = False

        if not foto1_acik:
            cv2.imshow("foto1",foto1)# FOTO1 İ AÇ
            foto1_acik = True

    #BAŞPARMAK EKRANA GÖRE 0.5 DEN AZSA FOTO1 AÇ
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
