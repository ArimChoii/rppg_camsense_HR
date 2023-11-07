import cv2
font = cv2.FONT_ITALIC
import numpy as np
import matplotlib.pyplot as plt

def faceDetect():
    eye_detect = False
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

    try:
        cap = cv2.VideoCapture('./vid-1.avi')
    except:
        print("video loading error")
        return

    detected_faces = []  # 얼굴 정보를 저장할 목록

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if eye_detect:
            info = "Eye Detection ON"
        else:
            info = "Eye Detection OFF"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Detected Face", (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)

            if eye_detect:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # 감지된 얼굴 정보를 저장
            detected_faces.append((x, y, w, h))

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30)

        if k == ord('i'):
            eye_detect = not eye_detect
        if k == 27:
            break

        # 얼굴 이미지를 창에 표시 (첫 번째 얼굴만)
        if detected_faces:
            (x, y, w, h) = detected_faces[0]
            face_image = frame[y:y + h, x:x + w]
            cv2.imshow("First Detected Face", face_image)

    cap.release()
    cv2.destroyAllWindows()

    for i, (x, y, w, h) in enumerate(detected_faces):
        print(f"Face {i + 1}: x={x}, y={y}, width={w}, height={h}")

        if i == 0:  # 첫 번째 얼굴만 저장
            face_image = frame[y:y + h, x:x + w]
            cv2.imwrite("first_detected_face.jpg", face_image)

faceDetect()

def ecg():
    file = open("./gt_1.txt", "rt")
    text = file.read()
    file.close()

    val = text.split()

    len(val)
    val[520]

    EcgS = {}

    for i, j in zip(['#I[uV]', '#II[uV]', '#III[uV]', '#avR[uV]', '#avL[uV]'],
                    ['#II[uV]', '#III[uV]', '#avR[uV]', '#avL[uV]', '#avF[uV]']):
        resf = np.array([_ for _ in range(len(val)) if val[_] == i])
        rese = np.array([_ for _ in range(len(val)) if val[_] == j])

        if len(resf)>0 and len(rese)>0:
            EcgS[i] = np.array(val[resf[0]+1: rese[0]]).astype(float)
    if j in EcgS:
        EcgS[j] = np.array(val[rese[0]+1]).astype(float)

    # plt.plot(val2)
    plt.plot(EcgS['#I[uV]'][8:]), plt.xlabel("Time"), plt.ylabel("Magnitude"), plt.title("ECG")
    type(EcgS)

ecg()