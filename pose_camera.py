#
# Author: SIANA Systems
#
# ref: https://github.com/google-coral/project-posenet
#

import platform
from datetime import datetime

import cv2
import numpy as np

import tflite_runtime
from pympcam.coralManager import CoralManager
from imutils.video import VideoStream, ImageOutput, FPS

from pose_engine import PoseEngine

#--->> TUNABLES <<-------------------------------------------------------------

verbose = False

config_level = 'low'

enable_web_output = True

#------------------------------------------------------------------------------

config = {
    'low':    [640,480,15,  'models/mobilenet/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite'],
    'medium': [640,480,15,  'models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'],
    'high':   [1280,720,15, 'models/mobilenet/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite']
}

model_file = ''
camera_width = 0
camera_height = 0
camera_fps = 0

# supported pose edges
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

def apply_config(option='low'):
    '''simple helper to select the proper configuration based on the specified option

      Args:
         option: either 'low','medium','high'
    '''      
    width = config[option][0]
    height = config[option][1]
    fps = config[option][2]
    file = config[option][3]
    return width,height,fps,file

def draw_pose(poses, src):
    global config_level, camera_width, camera_height
    '''Draws a stick-figure pose ontop of an input image.

      Args:
         poses: list of detected x/y key-points  
         scr:  input image
    '''
    threshold = 0.2

    if config_level == 'low':
        box_x, box_y, box_w, box_h = 5, 0, 470, 353 #inference_box
    elif config_level == 'medium':
        box_x, box_y, box_w, box_h = 5, 0, 630, 481 #inference_box
    else:
        box_x, box_y, box_w, box_h = 5, 0, 1270, 721 #inference_box

    scale_x, scale_y = camera_width / box_w, camera_height / box_h

    xys = {}

    # Draw circles
    for pose in poses:
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < threshold: continue
            # Offset and scale to source coordinate space.
            kp_y = int((keypoint.yx[0] - box_y) * scale_y)
            kp_x = int((keypoint.yx[1] - box_x) * scale_x)

            xys[label] = (kp_x, kp_y)
            
            cv2.circle(src, (int(kp_x), int(kp_y)), 5, (255, 0, 0), 1)

    # Draw lines
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        cv2.line(src, (ax, ay), (bx, by), (255, 0, 0), 2)

def draw_metrics(src, fps, model_msec=0):
    '''Draws the FPS & model duration on top of the input image.

       Args:
          src: the input image
          fps: the Frame-per-second value   
          model_msec: the model compute duration in msec
    '''
    canvas = np.zeros((20, camera_width, 3), np.uint8) + 255

    canvas = cv2.putText(canvas, 
                        'fps: {:.2f}, model: {:.0f} ms'.format(
                            float(round(fps,2)), 
                            float(round(model_msec))),
                        (15, 15),                 # bottom-left corner x/y 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6,                      # fontScale factor
                        (0,0,0),                  # color (r,g,b)
                        1,                        # thickness (px)
                        cv2.LINE_AA
                        )
                        
    return cv2.vconcat([src, canvas])        

def process_frame(img, model):
    '''Processes a new frame with the given model.

       Args:
          img: the input frame
          model: the TF model
    '''

    poses = model.DetectPosesInImage(img)
    if verbose: print("!! POSES:\n{}".format(poses))

    draw_pose(poses, img)
    return img

def main():
    global camera_width, camera_height, camera_fps, model_file 

    print("\n** MPCam: Coral PoseNet demo **\n")
    print(">> system info:")    
    print("\tpython {}".format(platform.python_version()))
    print("\ttflite {}".format(tflite_runtime.__version__))
    print("\topencv {}".format(cv2.__version__))

    print(">> configuration = {}".format(config_level))
    camera_width, camera_height, camera_fps, model_file = apply_config( config_level )
    
    print(">> turning on coral...")
    coral = CoralManager()
    coral.turnOn()

    print(">> configuring camera...")    
    camera = VideoStream()
    cv = camera.stream
    print("\tWidth/Height: {}/{}".format(camera_width, camera_height))
    cv.stream.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cv.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    print("\tFPS: {}".format(camera_fps))
    cv.stream.set(cv2.CAP_PROP_FPS, camera_fps)
    camera.start()

    print(">> loading model: {}".format(model_file))
    model = PoseEngine( model_file )    

    if enable_web_output:
        print(">> configuring web streamer: http://mpcam.local:8080"),
        http_display = ImageOutput(screen=False)
    else:
        print(">> web streamer is disabled!")
    
    print(">> starting processing...")
    try:
        last_print = datetime.now()
        # fps/time trackers
        global_fps = FPS().start()
        model_duration = 0
        nframes = 0
        # reported metrics
        report_fps = 0
        report_duration = 0
        while True:
            
            # grab frame
            frame = camera.read()            
            global_fps.update()
            nframes += 1
               
            # process frame          
            start_model = datetime.now()
            processed_frame = process_frame(frame, model)   
            model_duration += (datetime.now() - start_model).total_seconds()             

            # overlay metrics                
            processed_frame = draw_metrics(processed_frame, report_fps, report_duration)

            if enable_web_output:
                # output processed frame
                http_display.stream('posenet', processed_frame)
            
            # print metrics in console every 5sec
            if (datetime.now() - last_print).total_seconds() > 5:
                last_print = datetime.now()
                print(">> FPS: {:.2f}, Model: {:.0f} ms".format(report_fps, report_duration))                    

            # time to update metrics?
            if nframes > 10:
                # update global fps
                global_fps.stop()
                report_fps = global_fps.fps()
                # update averaged model duration
                report_duration = (model_duration / nframes) * 1000
                model_duration = nframes = 0

    finally:
        print(">> turn off camera & coral...")
        camera.stop()
        coral.turnOff()
        print(">> done!")

if __name__ == '__main__':
    main()