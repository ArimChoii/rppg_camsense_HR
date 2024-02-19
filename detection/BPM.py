import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.getcwd()
import numpy as np
import cv2

class FaceDetect:
    def __init__(self, video_path=None, sav_opt=0, filename=[]):
        self.video_path = video_path
        self.chick = []
        self.face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  #spped of frame
        self.sav_opt = sav_opt
        self.filename = filename
        self.red_values = []
        self.green_values = []
        self.blue_values = []
        self.time_values = []

    def run_vid(self):
        cropmn = []
        frame_count = 0
        i = 0
        pixel_values = []
        chickimg_for_time = []
        red = []
        green = []
        blue = []
        time_values = []

        while frame_count < int(10 * self.fps):
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                # face_cascade 이용한 얼굴 detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Detected Face", (x - 5, y - 5), cv2.FONT_ITALIC, 0.5, (255, 255, 0), 2)

                face_image = frame[y:y + h, x:x + w]
                cropmn.append(face_image)  # face_image 저장하는 리스트, cropmn 선언

                cv2.imshow("Detected face", face_image)

                [a, b, c, d] = [3 * w // 5, h // 2, 4 * w // 5, 2 * h // 3]  # 설정한 새로운 좌표
                cv2.rectangle(face_image, (a, b), (c, d), (255, 255, 255))

                chickimg = face_image[b:d, a:c]

                cv2.imshow("chickimg ", chickimg)

                # print('chickimg' : chickimg)
                self.chick.append(chickimg.mean())  # chick 리스트 선언

                # Calculate the center point of the chickimg rectangle
                center_x = 35 * w // 50  # Calculate center x-coordinate
                center_y = 25 * h // 60  # Calculate center y-coordinate

                # Get the pixel value at the center point
                pixel_value = face_image[center_y, center_x]  # Access pixel value
                # Append the pixel value to the list
                pixel_values.append(pixel_value)

                # cv2.imshow('sub_face', chickimg)

                # chickimg =  cv2.resize(chickimg, (200, 200)) # size of the plt
                if i < len(self.chick):
                    time_values.append(frame_count / self.fps)
                    time_values.append(frame_count / self.fps)
                    red.append(face_image[center_y, center_x][2])  # Red channel
                    green.append(face_image[center_y, center_x][1])  # Green channel
                    blue.append(face_image[center_y, center_x][0])  # Blue channel

                # Display the result
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1
            i += 1
            time_values.append(frame_count / self.fps)

        cv2.destroyAllWindows()

        pixd = {'cpm': cropmn, 'chickimg': self.chick}
        self.chick = np.array(self.chick)

        red_changes = np.diff(red)
        green_changes = np.diff(green)
        blue_changes = np.diff(blue)

        # RGB 값 변화량의 절대값의 합 계산
        total_changes = red_changes + green_changes + blue_changes
        #print("total_changes", total_changes)
        abs_total_changes = np.abs(total_changes)
        #print("abs_total_changes:", abs_total_changes)

        max_change_index = np.argmax(abs_total_changes[1:]) + 1
        print("max_change_index : ", max_change_index)
        time_in_seconds = max_change_index / self.fps
        print("time_in_seconds : ", time_in_seconds)
        if time_in_seconds == 0 or self.fps == 0:
            estimated_heart_rate = 0
        else:
            estimated_heart_rate = 60 / time_in_seconds

            # 대체값 설정
        if estimated_heart_rate == 0:
            estimated_heart_rate = 70


        if self.sav_opt:
            self.vidwrit(i2s=self.chick)
        # estimated_heart_rate 값이 0이면 다른 값으로 대체

        return estimated_heart_rate

        min_length = min(len(time_values), len(red), len(green), len(blue))
        time_values = time_values[:min_length]
        red_values = red[:min_length]
        green_values = green[:min_length]
        blue_values = blue[:min_length]

        # plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, red, label='Red', color='r')
        plt.plot(time_values, green, label='Green', color='g')
        plt.plot(time_values, blue, label='Blue', color='b')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pixel Value')
        plt.title('RGB Channels Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        return pixd, self.chick, pixel_values, red, blue, green, estimated_heart_rate

    def vidwrit(self, i2s=[]):
        out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (200, 200))
        for i in range(len(i2s)):
            out.write(i2s[i])
        out.release()

# 실제 영상을 실행하는 cell - vid 불러오기
vidname = ('./vid-1.avi')
cap = cv2.VideoCapture(vidname)
videop = FaceDetect(video_path=vidname, sav_opt=0, filename=vidname)
estimated_heart_rate = videop.run_vid()
print("vidname : ", vidname)
print(f"Estimated Heart Rate: {estimated_heart_rate} BPM")

cap.release()
cv2.destroyAllWindows()