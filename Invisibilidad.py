import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

time.sleep(3)
count = 0
background=0

#Esperamos a que la cámara se estabilice
for i in range(60):
	ret,background = cap.read()

while(cap.isOpened()):
	ret, img = cap.read()
	if not ret:
		break
	count+=1
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	min_rojo = np.array([0,120,70])
	max_rojo = np.array([20,255,255])
	mascara = cv2.inRange(hsv,min_rojo,max_rojo)

	min_rojo = np.array([160,120,70])
	max_rojo = np.array([180,255,255])
	mascara2 = cv2.inRange(hsv,min_rojo,max_rojo)

	mascara = mascara+mascara2

	mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
	mascara = cv2.dilate(mascara,np.ones((3,3),np.uint8),iterations = 1)
	mascara2 = cv2.bitwise_not(mascara)

	res1 = cv2.bitwise_and(background,background,mask=mascara)
	res2 = cv2.bitwise_and(img,img,mask=mascara2)
	img_final = cv2.addWeighted(res1,1,res2,1,0)

	cv2.imshow('Invisible',img_final)
	k = cv2.waitKey(10)
	if k == 27:
		break
