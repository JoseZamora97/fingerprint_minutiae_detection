import numpy as np
import cv2


class NormalizationUtils:

    @staticmethod
    def normalize(im, m0, v0):
        m = np.mean(im)
        c = np.sqrt((v0 * ((im - np.mean(im)) ** 2)) / (np.std(im) ** 2))
        return np.where(im > m, m0 + c, m0 - c).astype(np.uint8)

    @staticmethod
    def std_norm(im):
        return (im - np.mean(im)) / (np.std(im))

    @classmethod
    def get_fingerprint_mask(cls, im, block_size, threshold=.2):
        h, w = im.shape
        threshold = np.std(im) * threshold

        image_variance = np.zeros(im.shape)

        for i in range(0, w, block_size):
            for j in range(0, h, block_size):
                x0, y0, xf, yf = [i, j, min(i + block_size, w), min(j + block_size, h)]
                image_variance[y0:yf, x0:xf] = np.std(im[y0:yf, x0:xf])

        mask = np.ones_like(im)
        mask[image_variance < threshold] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size * 2, block_size * 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
