import numpy as np
from scipy import ndimage


class GaborFilter:

    @classmethod
    def gabor(cls, x, sigma_x, y, sigma_y, v):
        aux_x, aux_y = (x ** 2 / sigma_x ** 2), (y ** 2 / sigma_y ** 2)
        aux_cos = np.cos(2 * np.pi * v * x)

        # return np.exp(
        #     -((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y))) * np.cos(
        #     2 * np.pi * v * x)

        return np.exp(-(aux_x + aux_y)) * aux_cos

    @staticmethod
    def apply(im, orient, mask, kx=0.65, ky=0.65, alpha=0.11):

        angle_increment = 3
        im = np.double(im)
        h, w = im.shape
        return_img = np.zeros((h, w))

        # Generate filters corresponding to these distinct frequencies and
        # orientations in 'angle_increment' increments.
        sigma_x = 1 / alpha * kx
        sigma_y = 1 / alpha * ky

        block_size = np.round(3 * np.max([sigma_x, sigma_y]))

        array = np.linspace(-int(block_size), int(block_size), (2 * int(block_size) + 1))
        x, y = np.meshgrid(array, array)

        # gabor filter equation
        base_filter = GaborFilter.gabor(x, sigma_x, y, sigma_y, alpha)
        h_filter, w_filter = base_filter.shape
        gabor_filter = np.array(np.zeros((180 // angle_increment, h_filter, w_filter)))

        # Generate rotated versions of the filter.
        for degree in range(0, 180 // angle_increment):
            rotated_filter = ndimage.rotate(base_filter, -(degree * angle_increment + 90), reshape=False)
            gabor_filter[degree] = rotated_filter

        # Convert orientation matrix values from radians to an index value that
        # corresponds to round(degrees/angle_increment)
        max_orientation_index = np.round(180 / angle_increment)
        orientation_index = np.round(orient / np.pi * 180 / angle_increment)
        for i in range(0, h // 16):
            for j in range(0, w // 16):
                if orientation_index[i][j] < 1:
                    orientation_index[i][j] = orientation_index[i][j] + max_orientation_index
                if orientation_index[i][j] > max_orientation_index:
                    orientation_index[i][j] = orientation_index[i][j] - max_orientation_index

        # Find indices of matrix points greater than maxsze from the image boundary
        block_size = int(block_size)
        valid_row, valid_col = np.where(mask > 0)
        final_index = \
            np.where((valid_row > block_size) & (valid_row < h - block_size) & (valid_col > block_size) & (
                        valid_col < w - block_size))

        for k in range(0, np.shape(final_index)[1]):
            r = valid_row[final_index[0][k]]
            c = valid_col[final_index[0][k]]
            img_block = im[r - block_size:r + block_size + 1][:, c - block_size:c + block_size + 1]
            return_img[r][c] = np.sum(img_block * gabor_filter[int(orientation_index[r // 16][c // 16]) - 1])

        gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

        return gabor_img
