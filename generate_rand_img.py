import cv2
import numpy as np

img = cv2.imread("test/waterfall.jpg")

width = 32
height = 32

for i in range(50):
    y = int(np.random.random()*img.shape[0])
    x = int(np.random.random()*img.shape[1])
    if y + height < img.shape[0] and x + width <img.shape[1]:
        rand_img = np.copy(img[y:y+height,x:x+width])
        cv2.imwrite("{}{}{}".format("dataset/random/", i, ".png"), rand_img)
