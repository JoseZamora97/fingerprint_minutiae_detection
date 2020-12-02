import numpy as np
import cv2
from skimage import morphology as morph


class SkeletonUtils:

    @staticmethod
    def skeletonize(image_input, mask):
        """
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
        Skeletonization reduces binary objects to 1 pixel wide representations.
        get_skeletonization works by making successive passes of the image. On each pass,
        border pixels are identified and removed on the condition that they do not
        break the connectivity of the corresponding object.
        :param image_input: 2d array uint8
        :return:
        """
        image, output = np.zeros_like(image_input), np.zeros_like(image_input)
        image[image_input == 0] = 1.

        output[morph.skeletonize(image)] = 255
        cv2.bitwise_not(output, output)

        return (output * mask).astype(np.uint8)
