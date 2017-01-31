'''
[project objective here]
Authors: Nhung Hoang and Richard Phillips
Project Dates: 01/30/17 - 

Notes:
    Use argparse?
'''

import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def averaged_background(video, max_frames=600):
	"""
	Take a cv2 VideoCapture object and average out the frames over the whole video or the max_frames, whichever is smallest.
	:param video: cv2.VideoCapture
	:rtype: np.array
	"""
	# Get video dimensions
	width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	average_bg = np.zeros((height,width), dtype=np.float32)

	# Count frames to make sure we don't try to pull a frame that doesn't exist
	length = video.get(cv2.CAP_PROP_FRAME_COUNT)
	length = np.float64(min(length, max_frames))

	for i in range(int(length)):
		ret, frame = video.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		average_bg += np.array(gray)/length

	return average_bg

def threshold_frame(frame, background, threshold=20):
	"""
	Return a binary mask for a specific frame. Frame and background should be np.array or cv.mat objects.
	:return: np.array
	"""
	diff = np.array(frame, np.float32) - background
	ret, mask = cv2.threshold(diff,threshold, 255, cv2.THRESH_BINARY)
	# Debugging check. TODO: Remove in production.
	if np.max(mask) > 255.0:
		raise ValueError("Oops, the binary mask has written specific pixel values to be over 255.")
	return mask

# def morph_operate(frame):


def main(video_file=None):
	if video_file is None:
		video = cv2.VideoCapture(sys.argv[1])
	else:
		video = cv2.VideoCapture(video_file)
	bg = averaged_background(video)

	# plt.imshow(bg,cmap='gray')
	# plt.show()

	# Grab an example frame and threshold
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	frame = cv2.VideoCapture(video_file)
	## To Cut
	new_video = cv2.VideoWriter('newvid.mp4', fourcc, 20.0,(360,640),False)
	for i in range(1,50):
		ret, frame2 = frame.read()
		frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		frame3 = threshold_frame(frame2, bg, 35)
		print frame3.shape
		new_video.write(frame3)
		cv2.imshow('frame',frame3)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	# while video.isOpened():
	# 	ret,frame = video.read()
	# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 	cv2.imshow('frame',gray)
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break
	video.release()
	cv2.destroyAllWindows()



main("Clips/Part1.mov")
