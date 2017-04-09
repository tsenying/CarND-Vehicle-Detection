#
# main control code for processing video
#
import cv2
import pickle
import config
from vehicle_detection import VehicleDetection

# Read in the saved StandardScalar and trained classifier
# and feature detection settings
# These were saved by running "search_and_classify.py"
config.settings = pickle.load( open("svc_trained.p", "rb" ) )
print("config.settings={}".format(config.settings))
# dist_pickle = pickle.load( open("svc_trained.p", "rb" ) )

# file for logging debug info
config.debug_log = open('debug.log', 'w')

# open video file
cap = cv2.VideoCapture('test_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) # get frame per second info

# set up video writer for MP4
# http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
# *’mp4v’ -> .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cars_video = cv2.VideoWriter('cars_video.mp4',fourcc,fps,( config.image_shape['width'], config.image_shape['height'] ))

config.count = 0
config.vehicle_detection = VehicleDetection( config.nframes )

start_frame = 0 
stop_frame  = 1260 # total number of frames in video = 1260 

#
# For each frame:
#   process frame
#   write frame to video file
#
while(cap.isOpened()):
    # read next frame
    ret, frame = cap.read()
    if not ret:
        break
        
    config.count += 1
    if config.count%100 == 0:
        print("config.count={}".format(config.count))
        
    if start_frame <= config.count <= stop_frame:
        if config.count%50 == 0:
            print("processing frame {}, shape={}".format(config.count, frame.shape))
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./debug_images/orig/frame' + str(config.count) + '.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) )
        
        # process image
        image_with_cars = config.vehicle_detection.process_image(frame_rgb)
        #cv2.imwrite('./debug_images/cars/frame' + str(config.count) + '.jpg', cv2.cvtColor(image_with_cars, cv2.COLOR_RGB2BGR) )

        # write image to video
        cars_video.write( cv2.cvtColor(image_with_cars, cv2.COLOR_RGB2BGR) )
        
    if config.count > stop_frame:
        break

cap.release()
cv2.destroyAllWindows()
cars_video.release()

if config.debug_log is not None:
    config.debug_log.close()
    
print("total config.count={}".format(config.count))
