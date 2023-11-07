#경로를 영상있는 파일로 옮겨야됨

import os
import matplotlib.pyplot as plt
os.getcwd()
import numpy as np
import cv2
import numba
from numba import jit, cuda

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

a =[]
a.append(2)
a.append(5)
a
import glob

#########################
class vidproc:
    def __init__(self, cap=[], sav_opt=0, filename=[]):
        self.pixval1 = []
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.sav_opt = sav_opt
        self.filename = filename

    def run_vid(self):
        pixVal = []
        pixVal2 = []
        cropmn = []
        loop_t = 0
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret == 0:
                break
            # converting BGR to HSV

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            cropIm = frame[600:700, 700:750]
            cropmn.append(cropIm.mean())

            # eyes = eye_cascade.detectMultiScale(gray, 1.1,3)

            # for f in faces:
            # x, y, w, h = [ v for v in f ]
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255))

            # sub_face = frame[y-50:y+h+50, x:x+w]
            #
            # sub_face = frame[180:750, 750:1170]

            # sub_face = cv2.resize(sub_face, (600,700))

            # [a,b,c,d] = [30, 50, 550, 240]

            # cv2.rectangle(sub_face, (a,b), (c, d), (255,255,255))

            # chickimg = sub_face[b:d, a:c]

            # chickimg =  cv2.resize(chickimg, (200, 200))

            sub_face = frame[200:800, 670:1050]

            sub_face = cv2.resize(sub_face, (600, 700))

            [a, b, c, d] = [380, 350, 580, 550]

            cv2.rectangle(sub_face, (a, b), (c, d), (255, 255, 255))

            chickimg = sub_face[b:d, a:c]

            chickimg = cv2.resize(chickimg, (200, 200))

            # Display the resulting frame

            self.pixval1.append(chickimg)

            cv2.imshow('Video', sub_face)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            loop_t = +1

        pixd = {'p1': pixVal, 'p2': pixVal2, 'cpm': cropmn}
        self.pixval1 = np.array(self.pixval1)

        if self.sav_opt:
            self.vidwrit(i2s=self.pixval1)

        return pixd, self.pixval1
    pass

    def vidwrit(self, i2s=[]):

        out = cv2.VideoWriter('TrainPrep/RCheek/' + self.filename, cv2.VideoWriter_fourcc(*'DIVX'),
                              self.fps, (200, 200))
        for i in range(len(i2s)):
            out.write(i2s[i])
        out.release()
        pass

vidname = 'vid-1.avi'
cap = cv2.VideoCapture(vidname)
# Check if camera opened successfully
videop = vidproc(cap=cap, sav_opt=0, filename=vidname)
pixVal, pv1 = videop.run_vid()

cap.release()
cv2.destroyAllWindows()

pv1 = np.array(pv1)

cap = cv2.VideoCapture('vid-1.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(fps)