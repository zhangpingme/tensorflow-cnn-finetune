import glob
import os
import cv2

dir = "D:/data/pic20180505-1/step_1_train/0"
count = 0
img_dir = glob.glob(os.path.join(dir,"*.png"))
for i in img_dir:
    path = i[:48]
    name = path +"/" +str(count) +".png"
    os.rename(i, name)
    count+=1