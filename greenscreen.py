"""
comments comments comments
"""

import sys

import numpy as np
import cv2

def get_new_vids(vid, height, width):
    try:
      fourcc = cv2.cv.CV_FOURCC(*'XVID')
      masked_video = cv2.VideoWriter("masked_vid.avi", fourcc, vid.get(5), (width,height))
      morphed_video = cv2.VideoWriter("morphed_vid.avi", fourcc, vid.get(5), (width,height))
    except:    
      try:			
        fourcc, ext = (cv2.VideoWriter_fourcc(*'DIVX'), 'avi') 
        masked_video = cv2.VideoWriter("masked_vid.avi", fourcc, vid.get(5), (width,height))
        morphed_video = cv2.VideoWriter("morphed_vid.avi", fourcc, vid.get(5), (width,height))
      except:
        try:    
          fourcc = cv2.VideoWriter_fourcc(*'XVID')
          masked_video = cv2.VideoWriter("masked_vid.avi", fourcc, vid.get(5), (width,height))
          morphed_video = cv2.VideoWriter("morphed_vid.avi", fourcc, vid.get(5), (width,height))
        except:
          print "Three attempts at initializing fourcc failed. Check OS."
          sys.exit()
    return masked_video, morphed_video

def avg_background(video, max_frames = 40):
    """
    Take a cv2 VideoCapture object and average out the frames over the whole video or the max_frames, whichever is smallest.
    :param video: cv2.VideoCapture
    :rtype: np.array
    """
    # Get video dimensions
    width = int(video.get(3))
    height = int(video.get(4))
    print "w: ", width, "h: ", height
    average_bg = np.zeros((height,width,3), dtype=np.float64)

    # Count frames to make sure we don't try to pull a frame that doesn't exist
    duration = int(video.get(7))
    duration = np.float64(min(duration, max_frames))

    for i in range(int(duration)):
            ret, frame = video.read()
            average_bg += (np.array(frame)/duration)

    return average_bg

def blackout_bg(avg_bg, thres):
    vid = cv2.VideoCapture(sys.argv[1])
    dur = int(vid.get(7))
    print 'dur: ', dur
    height, width, ret = avg_bg.shape

		#write new video
    masked_video, morphed_video = get_new_vids(vid, height, width)

    for i in range(dur):
        masked_frame = np.zeros((height,width, 3))
        ret, frame = vid.read()
        dist = np.linalg.norm((avg_bg-frame),axis = 2)
        mask = dist > thres
        mask = mask[:,:,np.newaxis]
        masked_frame = mask*frame
        masked_video.write(np.uint8(masked_frame))
        kernel = np.ones((10,10), np.uint8)
        morphed_frame = cv2.erode(masked_frame, kernel, iterations = 2)
        morphed_video.write(np.uint8(morphed_frame))
    vid.release()
    masked_video.release()
    morphed_video.release()
    return masked_video, morphed_video

def main():
    video = cv2.VideoCapture(sys.argv[1])
    if video.isOpened():
        avg_bg = avg_background(video)
        video.release()
        masked_vid, morphed_vid = blackout_bg(avg_bg, 30)
    else:
        print "cannot open file"
        sys.exit()

main()
