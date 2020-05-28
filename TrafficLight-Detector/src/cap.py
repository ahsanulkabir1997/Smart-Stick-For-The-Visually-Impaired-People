import cv2
import time
camera = cv2.VideoCapture(0)
i = 0
while 0 < 10:
    return_value, image = camera.read()
    cv2.imshow('cap',image)
    i += 1
    cv2.waitKey(0)
del(camera)
