"""
comments comments comments
"""

import sys

import numpy as np
import cv2
import cvk2


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

    for i in range(120):
        ret, frame = vid.read()
    # for i in range(10):
    while True:
        masked_frame = np.zeros((height,width, 3))
        ret, frame = vid.read()
        if ret:
            dist = np.linalg.norm((avg_bg-frame),axis = 2)
            # mask = np.array([dist > thres,dist > thres,dist > thres])
            #mask = np.reshape(mask, (height, width, 3), order = 'F')
            #masked_frame = np.multiply(mask, frame)

            masked_frame[np.nonzero(dist>=thres)] = frame[np.nonzero(dist>=thres)]

            # masked_frame = np.uint8(masked_frame)
            # print frame[0][0]
            # print mask[0][0]
            # print masked_frame

            # cv2.imshow('masked', masked_frame)
            #
            # cv2.waitKey(0)

            masked_frame = np.array(masked_frame, dtype=np.float32)
            bw = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            ret1, morph_masked = cv2.threshold(bw, 27, 255, cv2.THRESH_BINARY)

            morph_masked = np.array(morph_masked, dtype=np.float32)
            morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
            # morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_CLOSE, np.ones((4,4),np.uint8))
            morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_CLOSE, np.ones((16,16),np.uint8))

            morph_masked = np.array((morph_masked),dtype=np.float32)

            import copy
            # ret1, morph_masked2 = cv2.threshold(np.uint8(morph_masked), 22, 255, cv2.THRESH_BINARY) #[1] because we just want the threshold, not the ret
            # print morph_masked2.max()
            # morph_masked= np.uint8(morph_masked)
            # morph_masked = np.uint8(morph_masked)
            # morph_masked = cv2.cvtColor(morph_masked, cv2.COLOR_RGB2GRAY)


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
                    cv2.drawContours( display, contours, j, ccolors[j % len(ccolors)], -1 )

                    # Compute some statistics about this contour.
                    info = cvk2.getcontourinfo(contours[j])

                    # Mean location and basis vectors can be useful.
                    mu = info['mean']
                    b1 = info['b1']
                    b2 = info['b2']

                    # Annotate the display image with mean and basis vectors.
                    cv2.circle( display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA )

                    # cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b1),
                    #           white, 1, cv2.LINE_AA )
                    #
                    # cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b2),
                    #           white, 1, cv2.LINE_AA )

                    (x1, y1, w1, h1) = cv2.boundingRect(cont)
                    cv2.rectangle(display, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                else:
                    # Contour area too small
                    pass




            cv2.imshow('masked', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            new_video.write(np.uint8(masked_frame))



    vid.release()
    new_video.release()
    return new_video



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


