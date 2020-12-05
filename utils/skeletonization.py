import numpy as np
import cv2
from skimage import morphology as morph


class SkeletonUtils:

    @staticmethod
    def skeletonize(image_input):
        image, output = np.zeros_like(image_input), np.zeros_like(image_input)
        image[image_input == 0] = 1.

        output[morph.skeletonize(image)] = 255
        cv2.bitwise_not(output, output)

        return output
