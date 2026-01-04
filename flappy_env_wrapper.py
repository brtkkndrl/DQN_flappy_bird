#
#   Enviroment wrapper.
#

import gymnasium as gym
import cv2
import numpy as np

class FlappyBirdImgObservation(gym.ObservationWrapper):
    """
        Observation wrapper for Flappy Bird enviroment.
        Returns grayscale image observation from original enviroment rendered frame.
    """
    def __init__(self, env):
        super().__init__(env)
        self._out_width = 72
        self._out_height = 106

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._out_height, self._out_width),
            dtype=np.uint8
        )

    def process_frame(self, rgb_array):
        cropped = rgb_array[:424, :, :] # crop
        hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0] / 179.0
        sat = hsv[:, :, 1] / 255.0

        hue_sat = 0.7 * hue + 0.3 * sat
        hue_sat = cv2.resize(hue_sat, (self._out_width, self._out_height), interpolation=cv2.INTER_NEAREST)
        hue_sat = (np.clip(hue_sat, 0.0, 1.0) * 255).astype(np.uint8)

        return hue_sat
    
    def process_frame_2(self, rgb_array):
        cropped = rgb_array[:424, :, :] # crop
        single_channel = cropped[:, :, 0]*0.21 + cropped[:, :, 1]*0.72 +cropped[:, :, 2]*0.07
        single_channel = cv2.resize(single_channel, (self._out_width, self._out_height), interpolation=cv2.INTER_NEAREST)
        return single_channel


    def observation(self, observation):
        rgb_array = self.render()
        return self.process_frame(rgb_array)