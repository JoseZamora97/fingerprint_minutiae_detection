import numpy as np
import cv2


class NormalizationUtils:

    @staticmethod
    def normalize_img(im, m0, v0):
        m = np.mean(im)
        c = np.sqrt((v0 * ((im - np.mean(im)) ** 2)) / (np.std(im) ** 2))
        return np.where(im > m, m0 + c, m0 - c).astype(np.uint8)

    @staticmethod
    def std_norm(im):
        return (im - np.mean(im)) / (np.std(im))

    @staticmethod
    def get_fingerprint_mask(im, W, threshold=.2):
        """
        Returns mask identifying the ROI. Calculates the standard deviation in each image block and max_distance the ROI
        It also normalises the intensity values of
        the image so that the ridge regions have zero mean, unit standard
        deviation.
        :param im: Image
        :param W: size of the block
        :param threshold: std max_distance
        :return: segmented_image
        """
        h, w = im.shape
        threshold = np.std(im) * threshold

        image_variance = np.zeros(im.shape)
        for i in range(0, w, W):
            for j in range(0, h, W):
                x0, y0, xf, yf = [i, j, min(i + W, w), min(j + W, h)]
                image_variance[y0:yf, x0:xf] = np.std(im[y0:yf, x0:xf])

        # apply max_distance
        mask = np.ones_like(im)
        mask[image_variance < threshold] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (W * 2, W * 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
