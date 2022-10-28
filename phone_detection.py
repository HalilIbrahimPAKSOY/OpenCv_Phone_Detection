#phone detection

import cv2
import numpy as np

img = cv2.imread('phone.jpg')
img = cv2.resize(img,(640,480))

phone_cascade = cv2.CascadeClassifier('cascade.xml')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

phones = phone_cascade.detectMultiScale(gray,1.1,5)

for (x,y,w,h) in phones:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('img',img)
#cv2.imshow('gray',gray)


cv2.waitKey(0)

cv2.destroyAllWindows()