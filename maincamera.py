import numpy as np
import cv2

from skimage import feature
from skimage import exposure
from imutils import paths

if __name__ == "__main__":
  print("hello python!")
  capture = cv2.VideoCapture('video/16.mp4')
  capture.set(3, 320)
  capture.set(4, 240)
  width = 32
  height = 32
  while(capture.isOpened()):
    _, frame = capture.read()
    frame = cv2.resize(frame, (640, 360), cv2.INTER_AREA)
    frame = frame[140:,:]
    scaleFactor = 1
    r = 0
    while r <= 4:
      resizedImg = cv2.resize(frame, (int(frame.shape[1]/scaleFactor), int(frame.shape[0]/scaleFactor)), interpolation=cv2.INTER_AREA)
      if(r == 0):
        scaleFactor+=.1
      else:
        scaleFactor*=scaleFactor
      r+=1
      h = 0
      while h+height <= resizedImg.shape[0]:
        w = 0
        while w+width <= resizedImg.shape[1]:
          cropedImg = np.copy(resizedImg[h:h+height,w:w+width])
          cropedImg = cv2.cvtColor(cropedImg, cv2.COLOR_BGR2GRAY)
          boxImage = np.copy(resizedImg)
          cv2.rectangle(boxImage, (w, h), (w+width, h+height), (0,255,0), 2)
          fdesc, HoGImg = feature.hog(cropedImg, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2), block_norm='L1', visualize=True)
          HoGImg = exposure.rescale_intensity(HoGImg, in_range=(0, 10))
          cv2.imshow("image", frame)
          cv2.imshow("resized", boxImage)
          cv2.imshow("HoG", HoGImg)
          cv2.imshow("crop",cropedImg)
          cv2.waitKey(1) 
          w+=4
        h+=4