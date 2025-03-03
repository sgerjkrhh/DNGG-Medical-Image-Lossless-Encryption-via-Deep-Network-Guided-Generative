import os
from scipy.ndimage import zoom
import cv2
import numpy as np

num = 80
size = 512 
src_dir = r'E:\hzt\bone\tumor\dataset\jpg_data\ct_tra'
dst_dir = r'secret_pics'

paths = os.listdir(src_dir)
np.random.shuffle(paths)

for path in paths[:num]:
    pic = cv2.imread(fr'{src_dir}\{path}', 0)
    H, W = pic.shape
    pic = zoom(pic, [size/H, size/W], order=3)
    cv2.imencode('.png', pic)[1].tofile(fr'{dst_dir}\{path}')

