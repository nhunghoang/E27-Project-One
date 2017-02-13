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
      final_video = cv2.VideoWriter("final_vid.avi", fourcc, vid.get(5), (width,height))
    except:
      try:
        fourcc, ext = (cv2.VideoWriter_fourcc(*'DIVX'), 'avi')
        masked_video = cv2.VideoWriter("masked_vid.avi", fourcc, vid.get(5), (width,height))
        morphed_video = cv2.VideoWriter("morphed_vid.avi", fourcc, vid.get(5), (width,height))
        final_video = cv2.VideoWriter("final_vid.avi", fourcc, vid.get(5), (width,height))
      except:
        try:
          fourcc = cv2.VideoWriter_fourcc(*'XVID')
          masked_video = cv2.VideoWriter("masked_vid.avi", fourcc, vid.get(5), (width,height))
          morphed_video = cv2.VideoWriter("morphed_vid.avi", fourcc, vid.get(5), (width,height))
          final_video = cv2.VideoWriter("final_vid.avi", fourcc, vid.get(5), (width,height))
        except:
          print "Three attempts at initializing fourcc failed. Check OS."
          sys.exit()
    return masked_video, morphed_video, final_video


def avg_background(video, max_frames = 40):
    """
    Take a cv2 VideoCapture object and average out the frames over the whole video or the max_frames, whichever is smallest.
    :param video: cv2.VideoCapture
    :rtype: np.array
    """
    video = cv2.VideoCapture(video)
    # Get video dimensions
    width = int(video.get(3))
    height = int(video.get(4))
    average_bg = np.zeros((height,width,3), dtype=np.float64)
    # Count frames to make sure we don't try to pull a frame that doesn't exist
    duration = int(video.get(7))
    duration = np.float64(min(duration, max_frames))
    for i in range(int(duration)):
            ret, frame = video.read()
            average_bg += (np.array(frame)/duration)
    video.release()
    return average_bg


def assign_points(point, moving_objects):
    nearest_index, nearest = 0, sys.maxint
    moving_objects = map(lambda x:x[1][-1], moving_objects)
    moving_objects = np.array(moving_objects)
    point = np.array(point)
    distances = np.sqrt(np.sum((moving_objects-point)**2,axis=1))
    return np.argmin(distances), np.min(distances)


def edit_video(orig_vid, old_bg, thres, new_bg=None):
    vid = cv2.VideoCapture(orig_vid)
    dur = int(vid.get(7))
    height, width, ret = old_bg.shape
    masked_video_writer, morphed_video_writer, final_video_writer = get_new_vids(vid, height, width)

    if new_bg != None:
      new_bg = cv2.VideoCapture(new_bg)
    
    moving_objects = []
    obj_lst_activity = {}

    # loop through video frames
    for i in range(dur):
        masked_frame = np.zeros((height,width,3))
        morphed_frame = np.zeros((height,width,3))
        final_frame = np.zeros((height,width,3))
        ret, frame = vid.read()

        # set up for Green Screen
        if new_bg != None:
            scene_change = np.zeros((height,width, 3), np.uint8)
            bgret, bgframe = new_bg.read()
            bgheight, bgwidth, bgret = bgframe.shape
            if not bgret:
                print("Background video not long enough!")
                bg = None

        if ret:
            # Thresholding
            dist = np.linalg.norm((old_bg-frame),axis = 2)
            masked_frame[np.nonzero(dist>=thres)] = frame[np.nonzero(dist>=thres)]
            masked_frame = np.array(masked_frame, dtype=np.float32)
            masked_video_writer.write(np.uint8(masked_frame))
            
            # Morphology
            bw = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            ret1, morph_masked = cv2.threshold(bw, 27, 255, cv2.THRESH_BINARY)
            morph_masked = np.array(morph_masked, dtype=np.float32)
            # Get rid of dots & speckles
            morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
            # Connect body parts
            morph_masked = cv2.morphologyEx(morph_masked,cv2.MORPH_CLOSE, np.ones((16,16),np.uint8))
            morph_masked = np.array((morph_masked), dtype=np.float32)
            morphed_frame = cv2.cvtColor(morph_masked, cv2.COLOR_GRAY2RGB)
            morphed_frame = morphed_frame/255 * frame
            morphed_video_writer.write(np.uint8(morphed_frame))

            # Green Screen         
            if new_bg != None:
                scene_change[np.nonzero(morph_masked)] = frame[np.nonzero(morph_masked)]
                scene_change[np.nonzero(0==morph_masked)] = bgframe[np.nonzero(0==morph_masked)]
                frame = scene_change

            # CCA and Tracking
            morph_masked2 = copy.deepcopy(np.array(morph_masked, dtype=np.uint8))
            try :
                image, contours, hierarchy = cv2.findContours(morph_masked2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                # Different opencv version! See & Cite!: 
                # http://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
                contours, hierarchy = cv2.findContours(morph_masked2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Following lines based on regions.py
            display = np.zeros((image.shape[0], image.shape[1], 3),
                      dtype='uint8')
            # The getccolors function from cvk2 supplies a useful list
            # of different colors to color things in with.
            ccolors = cvk2.getccolors()
            tracking = True
            if tracking:
                if len(contours) >= 1:
                    # Only map contours larger than at least 80% of the largest contour. Address this in outline/overview/pdf thingy.
                    max_area = max(map(lambda cont: cv2.contourArea(cont), contours))
                    for j, cont in enumerate(contours):
                        area = cv2.contourArea(cont)
                        # This helps us track the focus of our images.
                        if area >= 0.45 * max_area:
                            # Draw the contour as a colored region on the display image.
                            cv2.drawContours( display, contours, j, ccolors[1], -1 )
                            # Compute some statistics about this contour.
                            info = cvk2.getcontourinfo(contours[j])
                            # Mean location and basis vectors can be useful.
                            mu = np.array(np.round(info['mean']),dtype=int)
                            # If we are currently NOT tracking any moving objects
                            if len(moving_objects)==0:
                                # Start tracking this moving object
                                moving_objects.append([area,[mu]])
                                obj_lst_activity[len(moving_objects)-1] = 0
                            else:
                                # Find the closest point and the distance to that point
                                obj, dist = assign_points(mu, moving_objects)
                                # If it's not very close, it's pretty far away, or we haven't seen the object in a while
                                #  we'll treat it like a new object
                                if dist >= 100 or np.round(area/moving_objects[obj][0]) != 1 or obj_lst_activity[obj] >= 10:
                                    moving_objects.append([area,[mu]])
                                    obj_lst_activity[len(moving_objects)-1] = 0
                                else:
                                    # Otherwise, it's probably from the object with the nearest point in the last time step
                                    # Append the new position to that object's list of positions
                                    moving_objects[obj][0] = area
                                    moving_objects[obj][1].append(mu)
                                    obj_lst_activity[obj] = 0
                            # obj_lst_activity keeps track of the last time a point was added for an object.
                            # Age this time in the line below
                            obj_lst_activity = {key: entry + 1 for key, entry in obj_lst_activity.iteritems()}
                            # Drawing lines following objects
                            for obj,color in zip(range(len(moving_objects)),ccolors):
                                # If the object is still active, plot the line of its trajectory
                                if obj_lst_activity[obj] <= 10:
                                    # Enumerate through the object's past positions
                                    for index, place in enumerate(moving_objects[obj][1]):
                                        # Skip first index, as we don't know what came before that
                                        if index != 0:
                                            # Draw line between points in order. Extension of ball tracking idea found in
                                            # blog post at http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
                                            line_size = min(index, 12)
                                            cv2.line(frame, tuple(moving_objects[obj][1][index - 1]), tuple(moving_objects[obj][1][index]), color, line_size)
                                        # If we've been tracking this object for a while,
                                        # forget the earliest point we tracked it at
                                    if len(moving_objects[obj][1]) >= 25:
                                        moving_objects[obj][1] = moving_objects[obj][1][1:]
                            # Annotate the display image with mean and basis vectors.
                            cv2.circle( display, cvk2.array2cv_int(mu), 4, (0,0,0), 1, cv2.LINE_AA )
                            # Find the dimensions of a bounding rectangle around this connected component
                            (x1, y1, w1, h1) = cv2.boundingRect(cont)
                            # Draw said rectangle
                            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            if new_bg != None:
                final_video_writer.write(np.uint8(scene_change))
            else:
                final_video_writer.write(np.uint8(frame))
    vid.release()
    masked_video_writer.release()
    morphed_video_writer.release()
    final_video_writer.release()
    return masked_video_writer, morphed_video_writer, final_video_writer



def main():
    vid = int(raw_input("Welcome! Which video do you want to change?\n[1] One Person\n[2] Two People\n"))
    pref = int(raw_input("What do you want to do with your video?\n[1] Add threshold, morphology, CCA, and tracking to video\n[2] Perform 1 and change background to ocean\n[3] Perform 1 and change background to snow_mountain\n")) 
    if vid == 1:
        video = 'walking_down.mov'
    else:
        video = 'dan&nhung.mp4'
    avg_bg = avg_background(video)
    if pref == 1:
        masked_vid, morphed_vid, final_vid = edit_video(video, avg_bg, 30)
    if pref == 2:
        masked_vid, morphed_vid, final_vid = edit_video(video, avg_bg, 30, 'horrgopro_14fps.avi')
    if pref == 3:
        masked_vid, morphed_vid, final_vid = edit_video(video, avg_bg, 30, 'snow.mp4') 

main()


