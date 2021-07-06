# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pycoral.utils import edgetpu
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

from PIL import Image
import numpy as np
import cv2

#--->> TUNABLES <<-------------------------------------------------------------

verbose = False

EDGETPU_SHARED_LIB = 'libedgetpu.so.2'
POSENET_SHARED_LIB = 'posenet_lib/posenet_decoder.so'

#------------------------------------------------------------------------------

KEYPOINTS = (
  'nose',
  'left eye',
  'right eye',
  'left ear',
  'right ear',
  'left shoulder',
  'right shoulder',
  'left elbow',
  'right elbow',
  'left wrist',
  'right wrist',
  'left hip',
  'right hip',
  'left knee',
  'right knee',
  'left ankle',
  'right ankle'
)

class Keypoint:
    __slots__ = ['k', 'yx', 'score']

    def __init__(self, k, yx, score=None):
        self.k = k
        self.yx = yx
        self.score = score

    def __repr__(self):
        return 'Keypoint(<{}>, {}, {})'.format(self.k, self.yx, self.score)


class Pose:
    __slots__ = ['keypoints', 'score']

    def __init__(self, keypoints, score=None):
        assert len(keypoints) == len(KEYPOINTS)
        self.keypoints = keypoints
        self.score = score

    def __repr__(self):
        return 'Pose({}, {})'.format(self.keypoints, self.score)

class PoseEngine():
    """Engine used for pose tasks."""

    def __init__(self, model_path, mirror=False):
        """Creates a PoseEngine with given model.

        Args:
          model_path: String, path to TF-Lite Flatbuffer file.
          mirror: Flip keypoints horizontally.

        Raises:
          ValueError: An error occurred when model output is invalid.
        """
        if verbose: print("!! loading edgetpu delegate...")
        edgetpu_delegate = load_delegate(EDGETPU_SHARED_LIB)

        if verbose: print("!! loading posenet delegate...")
        posenet_decoder_delegate = load_delegate(POSENET_SHARED_LIB)
        
        if verbose: print("!! instanciating interpreter...")
        self._interpreter = Interpreter(model_path,
                                        experimental_delegates=[edgetpu_delegate, posenet_decoder_delegate])

        if verbose: print("!! allocating tensors...")
        self._interpreter.allocate_tensors()

        self._mirror = mirror

        self._input_tensor_shape = self.get_input_tensor_shape()
        if (self._input_tensor_shape.size != 4 or
                self._input_tensor_shape[3] != 3 or
                self._input_tensor_shape[0] != 1):
            raise ValueError(
                ('Image model should have input shape [1, height, width, 3]!'
                 ' This model has {}.'.format(self._input_tensor_shape)))

        _, self._input_height, self._input_width, self._input_depth = self.get_input_tensor_shape()
        if verbose: print("!! mobel.input size => H = {}, W = {}, D = {}".format(self._input_height, self._input_width, self._input_depth))

        self._input_type = self._interpreter.get_input_details()[0]['dtype']
        if verbose: print("!! model.input type = {}".format(self._input_type))

    def DetectPosesInImage(self, img):
        """Detects poses in a given image.

           For ideal results make sure the image fed to this function is close to the
           expected input size - it is the caller's responsibility to resize the
           image accordingly.

        Args:
          img: numpy array containing image
        """

        # resize image to fit model...
        resized_image = cv2.resize(img, (self._input_width, self._input_height), Image.NEAREST)
        input_data = np.asarray(resized_image)

        # run inference + parser...
        edgetpu.run_inference(self._interpreter, input_data.flatten())
        return self.ParseOutput()

    def get_input_tensor_shape(self):
        """Returns input tensor shape."""
        return self._interpreter.get_input_details()[0]['shape']

    def get_output_tensor(self, idx):
        """Returns output tensor view."""
        return np.squeeze(self._interpreter.tensor(
            self._interpreter.get_output_details()[idx]['index'])())

    def ParseOutput(self):
        """Parses interpreter output tensors and returns decoded poses."""

        # extract output data...
        keypoints = self.get_output_tensor(0)
        keypoint_scores = self.get_output_tensor(1)
        pose_scores = self.get_output_tensor(2)
        num_poses = self.get_output_tensor(3)

        # holds the resulting poses
        poses = []

        # for each detected pose...
        for pose_i in range(int(num_poses)):
            pose_score = pose_scores[pose_i]            

            # process all pose-keypoints...
            keypoint_dict = {}
            for point_i, point in enumerate(keypoints[pose_i]):
                keypoint = Keypoint(KEYPOINTS[point_i], point, keypoint_scores[pose_i, point_i])
                if self._mirror: keypoint.yx[1] = self._input_width - keypoint.yx[1]
                keypoint_dict[ KEYPOINTS[point_i] ] = keypoint

            poses.append( Pose(keypoint_dict, pose_score) )

        return poses