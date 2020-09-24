import numpy as np
import cv2

from skimage import feature
from skimage import exposure
from imutils import paths

cntPt = []
click = False
mouseDragging = False
def RoI_pos(event, x, y, flags, param):
  global cntPt, click, mouseDragging
  if (event == cv2.EVENT_LBUTTONDOWN or mouseDragging):
    cntPt = [x, y]
    click = True
    mouseDragging = True
  if(event == cv2.EVENT_LBUTTONUP):
    click = False
    mouseDragging = False
  pass

if __name__ == "__main__":
  print("welcome to dataset creator :)")
  print("create ROI in every frame and save it")
  video = cv2.VideoCapture('10.mp4')
  cv2.namedWindow("stillframe")
  cv2.setMouseCallback("stillframe", RoI_pos)
  reading = True
  w = 0
  h = 0
  maxW = 640
  maxH = 360
  nscale = 0
  numCaptured = 0
  while reading:
    _, frame = video.read()
    while True:
      stillframe = np.copy(frame)
      stillframe = cv2.resize(stillframe, (maxW, maxH), cv2.INTER_AREA)
      croppedImg = np.copy(stillframe[h:h+32,w:w+32])
      croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
      cv2.rectangle(stillframe, (w, h), (w+32, h+32), (221,185,65), 1)
      fdesc, HoGImg = feature.hog(croppedImg, orientations=9, pixels_per_cell=(4,4),cells_per_block=(2,2), block_norm='L1', visualize=True)
      HoGImg = exposure.rescale_intensity(HoGImg, in_range=(0, 10))
      np.uint8(HoGImg)
      cv2.imshow("stillframe", stillframe)
      cv2.imshow("croppedImg", croppedImg)
      cv2.imshow("HoGImg", HoGImg)
      ch = cv2.waitKey(5)
      # if(ch != -1):
      #   print(ch)
      if(ch & 0xFF == ord('q')):
        reading = False
        break
      elif(ch & 0xFF == ord('n')):
        break
      elif(ch & 0xFF == ord('a')):
        if(nscale > -4 and nscale <= 0):
          maxW = int(float(maxW)/1.1)
          maxH = int(float(maxH)/1.1)
        if(nscale <= -4): nscale = -3
        nscale-=1
      elif(ch & 0xFF == ord('r')):
        maxW = 640
        maxH = 360
        nscale = 0
      elif(ch & 0xFF == ord('c')):
        print("%04d.png captured!"%(numCaptured))
        cv2.imwrite("dataset/%04d.png"%(numCaptured), croppedImg)
        numCaptured+=1
      elif(click):
        w = cntPt[0] - 16
        h = cntPt[1] - 16
        if(w<0): w = 0
        if(h<0): h = 0
        if(w+32>maxW): w = maxW - 32
        if(h+32>maxH): h = maxH - 32

  video.release()
  pass