#
# ref: https://github.com/google-coral/project-posenet
#

import time
import platform
from datetime import datetime
from io import StringIO

import cv2
import numpy as np
from PIL import Image

#from pympcam.coralManager import CoralManager

from pose_engine import PoseEngine
from imutils.video import VideoStream, ImageOutput, FPS

from pkg_resources import parse_version
from edgetpu import __version__ as edgetpu_version
assert parse_version(edgetpu_version) >= parse_version('2.11.1'), \
        'This demo requires Edge TPU version >= 2.11.1'

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

def draw_pose(poses, src):
    '''Draws a stick-figure pose ontop of an input image.

      Args:
         poses: list of detected x/y key-points  
         scr:  input image
    '''
    threshold = 0.2
    src_size = (480, 360)
    box_x, box_y, box_w, box_h = 5, 0, 470, 353 #inference_box

    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h

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
    canvas = np.zeros((20, 480, 3), np.uint8) + 255

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
    
    #img = cv2.flip(img, 1)
    img = cv2.resize(img, (480, 360))

    poses, _ = model.DetectPosesInImage(img)
    draw_pose(poses, img)
    return img

def main():

    print("\n** MPCam: Coral PoseNet demo **\n")
    print(">> system info:")
    print("\tpython {}".format(platform.python_version()))
    print("\tedgetpu {}".format(edgetpu_version))
    print("\topencv {}".format(cv2.__version__))
    
    print(">> turning on coral...")
    #coral = CoralManager()
    #coral.turnOn()
    time.sleep(2)

    print(">> configuring camera...")
    camera = VideoStream()
    cv = camera.stream
    cv.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cv.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv.stream.set(cv2.CAP_PROP_FPS, 15)
    camera.start()

    print(">> configuring web streamer: http://mpcam.local:8080"),
    http_display = ImageOutput(screen=False)

    print(">> loading model...")
    model = PoseEngine('models/mobilenet/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')
    
    print(">> starting processing...")
    try:
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

            # output processed frame
            http_display.stream('posenet', processed_frame)

            # time to update metrics?
            if nframes > 10:
                # update global fps
                global_fps.stop()
                report_fps = global_fps.fps()
                # update model duration
                report_duration = (model_duration / nframes) * 1000
                model_duration = nframes = 0

    except KeyboardInterrupt:
        fps.stop()
        print(">> turn off camera & coral...")
        camera.stop()
        coral.turnOff()
        print(">> done!")


if __name__ == '__main__':
    main()