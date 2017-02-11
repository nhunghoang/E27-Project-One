"""
comments comments comments
"""

import sys

import numpy as np
import cv2

<<<<<<< HEAD
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

=======
>>>>>>> f58e2b0b35fcc37580e9db20133443656553edd7
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
<<<<<<< HEAD
=======
    #length = video.get(cv2.CV_CAP_PROP_FRAME_COUNT)
>>>>>>> f58e2b0b35fcc37580e9db20133443656553edd7
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

<<<<<<< HEAD
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
=======
		#fourcc options

    fourcc = 0
    try:
      fourcc = cv2.cv.CV_FOURCC(*'XVID')
      new_video = cv2.VideoWriter("blackout_vid.avi", fourcc, vid.get(5)/10, (width,height))
    except:    
      try:			
        fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi') 
        new_video = cv2.VideoWriter("blackout_vid.avi", fourcc, vid.get(5)/10, (width,height))
      except:
        try:    
          fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
          new_video = cv2.VideoWriter("blackout_vid.avi", fourcc, vid.get(5)/10, (width,height))
        except:
          print "Three attempts are initializing fourcc failed. Check OS."
          sys.exit()

    for i in range(150):
        ret, frame = vid.read()
    for i in range(10):
    #for i in range(dur):
        masked_frame = np.zeros((height,width, 3))
        ret, frame = vid.read()
        dist = np.linalg.norm((avg_bg-frame),axis = 2)
        mask = np.array([dist > thres,dist > thres,dist > thres])
        #mask = np.reshape(mask, (height, width, 3), order = 'F')
        #masked_frame = np.multiply(mask, frame)
        
        #print frame[0][0]
        #print mask[0][0]
        #print masked_frame[0][0]
        #cv2.imshow('masked', masked_frame)
        #cv2.waitKey(0)
        
        for h in range(height):
            for w in range(width):
                if (dist[h][w] < thres):
                    masked_frame[h][w] = np.array([0,0,0])
                else:
                    masked_frame[h][w] = frame[h][w]
        print "frame number: ", i
        #cv2.imshow('masked frame', masked_frame)
        #cv2.waitKey(0)
            
        new_video.write(np.uint8(masked_frame))

    vid.release()
    new_video.release()
    return new_video
>>>>>>> f58e2b0b35fcc37580e9db20133443656553edd7

def main():
    video = cv2.VideoCapture(sys.argv[1])
    if video.isOpened():
        avg_bg = avg_background(video)
        video.release()
<<<<<<< HEAD
        masked_vid, morphed_vid = blackout_bg(avg_bg, 30)
=======
        masked_vid = blackout_bg(avg_bg, 30)
        
>>>>>>> f58e2b0b35fcc37580e9db20133443656553edd7
    else:
        print "cannot open file"
        sys.exit()

main()
