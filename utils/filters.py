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
    def apply(im, orient, mask, kx=0.65, ky=0.65, alpha=0.11):
        """
        Gabor filter is a linear filter used for edge detection. Gabor filter can be viewed as a sinusoidal plane of
        particular frequency and orientation, modulated by a Gaussian envelope.
        :param im:
        :param orient:
        :param freq:
        :param kx:
        :param ky:
        :return:
        """
        angleInc = 3
        im = np.double(im)
        rows, cols = im.shape
        return_img = np.zeros((rows, cols))
        freq = 0.1
        unfreq = 0.1

        # Generate filters corresponding to these distinct frequencies and
        # orientations in 'angleInc' increments.
        sigma_x = 1 / unfreq * kx
        sigma_y = 1 / unfreq * ky
        block_size = np.round(3 * np.max([sigma_x, sigma_y]))
        array = np.linspace(-int(block_size), int(block_size), (2 * int(block_size) + 1))
        x, y = np.meshgrid(array, array)

        # gabor filter equation
        reffilter = np.exp(
            -((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y))) * np.cos(
            2 * np.pi * unfreq * x)
        filt_rows, filt_cols = reffilter.shape
        gabor_filter = np.array(np.zeros((180 // angleInc, filt_rows, filt_cols)))

        # Generate rotated versions of the filter.
        for degree in range(0, 180 // angleInc):
            rot_filt = ndimage.rotate(reffilter, -(degree * angleInc + 90), reshape=False)
            gabor_filter[degree] = rot_filt

        # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angleInc)
        maxorientindex = np.round(180 / angleInc)
        orientindex = np.round(orient / np.pi * 180 / angleInc)
        for i in range(0, rows // 16):
            for j in range(0, cols // 16):
                if orientindex[i][j] < 1:
                    orientindex[i][j] = orientindex[i][j] + maxorientindex
                if orientindex[i][j] > maxorientindex:
                    orientindex[i][j] = orientindex[i][j] - maxorientindex

        # Find indices of matrix points greater than maxsze from the image boundary
        block_size = int(block_size)
        valid_row, valid_col = np.where(mask > 0)
        finalind = \
            np.where((valid_row > block_size) & (valid_row < rows - block_size) & (valid_col > block_size) & (
                        valid_col < cols - block_size))

        for k in range(0, np.shape(finalind)[1]):
            r = valid_row[finalind[0][k]]
            c = valid_col[finalind[0][k]]
            img_block = im[r - block_size:r + block_size + 1][:, c - block_size:c + block_size + 1]
            return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r // 16][c // 16]) - 1])

        gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

        return gabor_img
