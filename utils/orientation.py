import math

import cv2
import numpy as np


class OrientationUtils:

    @staticmethod
    def _get_line_ends(i, j, W, tang):
        if -1 <= tang <= 1:
            begin = (i, int((-W / 2) * tang + j + W / 2))
            end = (i + W, int((W / 2) * tang + j + W / 2))
        else:
            begin = (int(i + W / 2 + W / (2 * tang)), j + W // 2)
            end = (int(i + W / 2 - W / (2 * tang)), j - W // 2)
        return begin, end

    @staticmethod
    def visualize_angles(mask, angles, W):
        h, w = mask.shape
        result = cv2.cvtColor(np.zeros(mask.shape, np.uint8), cv2.COLOR_GRAY2RGB)
        mask_threshold = (W - 1) ** 2

        for i in range(1, w, W):
            for j in range(1, h, W):
                radian = np.sum(mask[j - 1:j + W, i - 1:i + W])
                if radian > mask_threshold:
                    tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                    begin, end = OrientationUtils._get_line_ends(i, j, W, tang)
                    cv2.line(result, begin, end, color=150)

        cv2.resize(result, mask.shape, result)

        return result

    @staticmethod
    def calculate_angles(im, W):
        """
        anisotropy orientation estimate, based on equations 5 from:
        https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf
        :param im:
        :param W: int W of the ridge
        :return: array
        """
        h, w = im.shape

        gx = cv2.Sobel(im / 125, -1, 1, 0) * 125
        gy = cv2.Sobel(im / 125, -1, 0, 1) * 125

        angles = [[] for _ in range(1, h, W)]

        for j in range(1, h, W):
            for i in range(1, w, W):

                u = slice(j, min(j + W, h - 1))
                v = slice(i, min(i + W, w - 1))

                nominator = np.sum(2 * sum(x * y for x, y in zip(gx[u, v], gy[u, v])).ravel())
                denominator = np.sum((gx[u, v] ** 2 - gy[u, v] ** 2).ravel())

                if nominator or denominator:
                    angle = (math.pi + math.atan2(nominator, denominator)) / 2
                    angles[int((j - 1) // W)].append(angle)
                else:
                    angles[int((j - 1) // W)].append(0)

        return np.array(angles)
