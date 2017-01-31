'''
[project objective here]
Authors: Nhung Hoang and Richard Phillips
Project Dates: 01/30/17 - 
'''

import numpy as np
import cv2
import sys

def main():
	video = cv2.VideoCapture(sys.argv[1])
	while video.isOpened():
		ret,frame = video.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video.release()
	cv2.destroyAllWindows()
main()
