import os
import glob

dir = "D:/data/pic20180505-1/step_1_test"
img_dir = os.path.join(dir,"*/*")
img_list = glob.glob(img_dir)
file_name = "../data/val.txt"
f = open(file_name,"w")

for i in img_list:
    if i[34:35] == "0":
        name = i + " 0\n"
    else:
        name = i + " 1\n"
    f.write(name)
