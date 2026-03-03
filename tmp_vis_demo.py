import os
import h5py
import natsort
import matplotlib.pyplot as plt

files = natsort.humansorted(os.listdir("/media/collab/T7/CoData/pi0_library/train"))
print(files)
for file in files:
    with h5py.File(f"/media/collab/T7/CoData/pi0_library/train/{file}", "r") as f:
        print(file)
        print(f.attrs["task_description"])
        image = f['obs']['left_realsense_img_rgb'][0]
        plt.imshow(image)
        plt.show(block=False)
        input("*************************")
        
