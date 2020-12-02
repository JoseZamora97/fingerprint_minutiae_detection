import cv2
import numpy as np
from scipy import ndimage


class GaborFilter:

    @staticmethod
    def gabor(x, sigma_x, y, sigma_y, v):
        aux_x, aux_y = (x ** 2 / sigma_x ** 2), (y ** 2 / sigma_y ** 2)
        aux_cos = np.cos(2 * np.pi * v * x)

        return np.exp(-(aux_x + aux_y)) * aux_cos

    @staticmethod
    def apply(im, angles, mask, kx=0.65, ky=0.65, alpha=0.11):
        """
        Gabor filter is a linear filter used for edge detection.
        Gabor filter can be viewed as a sinusoidal plane of
        particular frequency and orientation, modulated by a Gaussian envelope.
        """
        im_ = np.double(im)

        # Generate filters corresponding to these distinct frequencies and
        # orientations in 'angle_inc' increments.
        sigma_x = 1 / alpha * kx
        sigma_y = 1 / alpha * ky

        block_size = 16
        kernel_size = 20
        array = np.linspace(-kernel_size, kernel_size, (2 * kernel_size + 1))
        x, y = np.meshgrid(array, array)

        gabor = GaborFilter.gabor(x, sigma_x, y, sigma_y, alpha)

        im_out = im.copy()
        h_a, w_a = angles.shape

        for i in range(h_a):
            for j in range(w_a):
                degree = -(angles[i, j] * 180 / np.pi - 90)
                gabor_rotated_filter = ndimage.rotate(gabor, degree, reshape=False, mode='nearest')
                x0, xf, y0, yf = [
                    i * block_size, i * block_size + block_size,
                    j * block_size, j * block_size + block_size
                ]
                im_out[x0:xf, y0:yf] = cv2.filter2D(im_[x0:xf, y0:yf], -1, gabor_rotated_filter)

        # im_out[im_out < 0] = 0
        # im_out_ = (255 * (im_out - np.min(im_out)) / (np.max(im_out) - np.min(im_out))).astype(np.uint8)

        return 255 - (255 * (im_out - np.min(im_out)) / (np.max(im_out) - np.min(im_out)))
