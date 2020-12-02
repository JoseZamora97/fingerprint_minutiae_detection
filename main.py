import cv2

import utils
from crop import plot

# input_img = cv2.imread('./fingerprints_cropped/101_1.tif', cv2.COLOR_BGR2GRAY)
input_img = cv2.imread('./database/fingerprints/101_2.tif', cv2.COLOR_BGR2GRAY)
block_size = 16

# normalization -> orientation -> frequency -> mask -> filtering
# normalization - removes the effects of sensor noise and finger pressure differences.
im_normalized = utils.NormalizationUtils.normalize_img(input_img.copy(), float(100), float(100))

# ROI and normalisation
mask = utils.NormalizationUtils.get_fingerprint_mask(im_normalized, block_size, 0.2)
im_segmented = mask * im_normalized
im_norm = utils.NormalizationUtils.std_norm(im_normalized)

# orientations
im_angles = utils.OrientationUtils.calculate_angles(im_norm, W=block_size)
im_orientation = utils.OrientationUtils.visualize_angles(mask, im_angles, W=block_size)

# create gabor filter and do the actual filtering
gabor_img = utils.GaborFilter.apply(im_norm, im_angles, mask)

# thinning oor get_skeletonization
thin_image = utils.SkeletonUtils.skeletonize(gabor_img, mask)

# minutiae
minutiae = utils.MinutiaeUtils.calculate_minutiae(thin_image)

plot(minutiae, "")
