"""
comments comments comments
"""
import copy
import sys

import numpy as np
import cv2
import cvk2

def get_new_vids(vid, height, width):
    ''' Create two new videos. There are three ways to obtain fourcc, dependent on your OS.
        @PARAMS vid - the original video which to base the two new videos off of
								height/width - dimensions of the original video, passed down to the new videos
        @RETURN a video to hold the masked (threshold) format, a video to hold the morphed format
   '''
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
    #length = video.get(cv2.CV_CAP_PROP_FRAME_COUNT)
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

    masked_video_writer, morphed_video_writer = get_new_vids(vid, height, width)


    for i in range(120):
        ret, frame = vid.read()
    # for i in range(10):
    while True:
        masked_frame = np.zeros((height,width, 3))
        ret, frame = vid.read()
        if ret:
            dist = np.linalg.norm((avg_bg-frame),axis = 2)

            masked_frame[np.nonzero(dist>=thres)] = frame[np.nonzero(dist>=thres)]

            masked_frame = np.array(masked_frame, dtype=np.float32)
            bw = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            ret1, morph_masked = cv2.threshold(bw, 27, 255, cv2.THRESH_BINARY)

            # Morphology stuff
            morph_masked = np.array(morph_masked, dtype=np.float32)
            # Get rid of dots & speckles
            morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
            # Connect body parts
            morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_CLOSE, np.ones((16,16),np.uint8))
            morph_masked = np.array((morph_masked),dtype=np.float32)

            # CCA to find contours
            morph_masked2 = copy.deepcopy(np.array(morph_masked, dtype=np.uint8))
            try :
                image, contours, hierarchy = cv2.findContours(morph_masked2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            except:
                # Different opencv version! See & Cite!: http://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
                contours, hierarchy = cv2.findContours(morph_masked2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # Following lines based on regions.py
            display = np.zeros((image.shape[0], image.shape[1], 3),
                      dtype='uint8')

            # The getccolors function from cvk2 supplies a useful list
            # of different colors to color things in with.
            ccolors = cvk2.getccolors()

            # Define the color white (used below).
            white = (255,255,255)

            # Only map contours larger than at least 80% of the largest contour. Address this in outline/overview/pdf thingy.
            max_area = max(map(lambda cont: cv2.contourArea(cont), contours))

            for j, cont in enumerate(contours):
                area = cv2.contourArea(cont)
                if area >= 0.75 * max_area:
                    # Draw the contour as a colored region on the display image.
                    cv2.drawContours( display, contours, j, ccolors[1], -1 )

                    # Compute some statistics about this contour.
                    info = cvk2.getcontourinfo(contours[j])

                    # Mean location and basis vectors can be useful.
                    mu = info['mean']
                    b1 = info['b1']
                    b2 = info['b2']

                    # Annotate the display image with mean and basis vectors.
                    cv2.circle( display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA )

                    (x1, y1, w1, h1) = cv2.boundingRect(cont)
                    cv2.rectangle(display, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                else:
                    # Contour area too small
                    pass




            cv2.imshow('masked', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            masked_video_writer.write(np.uint8(masked_frame))



    vid.release()
    masked_video_writer.release()
    return masked_video_writer



def main():
    video = cv2.VideoCapture(sys.argv[1])
    if video.isOpened():
        avg_bg = avg_background(video)
        video.release()
        masked_vid = blackout_bg(avg_bg, 30)

        # video = cv2.VideoCapture(sys.argv[1])
        # print type(video)
        # frame = video
        # while True:
        #     ret, frame2 = frame.read()
        #     if not ret:
        #         break
        #     else:
        #         # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        #         # new_video.write(frame3)
        #         cv2.imshow('frame',frame2)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

    else:
        print "cannot open file"
        sys.exit()

sys.argv.append('walking_down.mov')
main()


