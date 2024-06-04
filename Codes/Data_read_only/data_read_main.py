import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # CNN overload error
import numpy as np
import cv2
import glob
import pandas as pd

#%%
path_dir = "C:\\SPB_Data\\project_rppg\\subject\\subject3"
dataPath = os.path.join(path_dir, '*.avi')
files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
files.sort()
list.sort(files) # serialing the data

print("Files:", files)
print("Video File Path:", files[0])

if not files:
    print("No bideo files found in the specified directory")
else:
    data=[]
    im_size=(100,100)
#%% Load the Video and corresponding GT Mat file
import pdb
cap = cv2.VideoCapture(files[0])

print("cap", cap)
if not cap.isOpened():
    print("Error: Video file could not be opened.")
#이건 내가 추가한 것

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = gray[:,:,1]
    #update the following line
    gray = gray[10:490, 80:720]
    gray = cv2.resize(gray, im_size)

    # Adjust the cropping and resizing for a final size of 640x480
    cropped_gray = gray[10:490, 80:720]  # Crop to match 640x480 aspect ratio
    resized_gray = cv2.resize(cropped_gray, (1280, 960))

    # pdb.set_trace()
    data.append(gray)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #추가...?
    if gray.shape[0] > 0 and gray.shape[1] > 0:
        gray = cv2.resize(gray, im_size)
        data.append(gray)
        cv2.imshow('frame', gray)
    else:
        print("Error: Source image is empty.")
    im_size = (100, 100)  # Example size; make sure it's valid

fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()
data =  np.array(data)

#%% PPG signal selection and alignment. 

# The starting points are the crucial, 
# this section needs select both the sratrting of video and the ppg point
# check fps and starting time in BVP.csv
# Match the lines from the supplimentary text file for the data