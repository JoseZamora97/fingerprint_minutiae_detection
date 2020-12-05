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

        block_size = int(block_size)
        im_padded = np.pad(im, [block_size, block_size], mode='constant')

        img_result = np.zeros((h, w))

        for r in range(block_size, im.shape[0]):
            for c in range(block_size, im.shape[1]):
                img_block = im_padded[r - block_size:r + block_size + 1][:, c - block_size:c + block_size + 1]
                img_result[r - block_size][c - block_size] = np.sum(img_block * gabor_filter[int(orientation_index[r // 16 - 1][c // 16 - 1]) - 1])

        gabor_img = 255 - np.array((img_result < 0) * 255).astype(np.uint8)

        return 255 - gabor_img * mask
