import csv
import os
from glob import glob

import cv2
from tqdm import tqdm

import utils

# Set up the params
fingerprints_path = './fingerprints'
output_path = './results'
block_size = 16  # Completely tested with 16 (square 16x16 for calculation)
max_distance = 8  # Threshold to consider a match
# Create the directory where all will be saved
os.makedirs(output_path, exist_ok=True)
# Open the CSV files
csv_file_matches = open(f"{output_path}/results_matches.csv", "w")
csv_file_full_matches = open(f"{output_path}/results_full_matches.csv", "w")
csv_file_misses = open(f"{output_path}/results_misses.csv", "w")
# Open the CSV writers
csv_writer_matches = csv.writer(csv_file_matches)
csv_writer_full_matches = csv.writer(csv_file_full_matches)
csv_writer_misses = csv.writer(csv_file_misses)
# Write the CSV headers
csv_writer_matches.writerow(["file"] + [e for e in utils.FingerprintAnalyzer.header_matches])
csv_writer_full_matches.writerow(["file"] + [e for e in utils.FingerprintAnalyzer.header_full_matches])
csv_writer_misses.writerow(["file"] + [e for e in utils.FingerprintAnalyzer.header_misses])
# Iterate over the whole path finding .tif images
for filepath in tqdm(glob(f"{fingerprints_path}/*.tif"), desc='Analysing Fingerprints', position=0):
    # Extract the fingerprint name
    fingerprint_name = os.path.splitext(os.path.basename(filepath))[0]
    path_fingerprint = f"{output_path}/{fingerprint_name}"
    # Create the folder to store the fingerprint analysis results
    os.makedirs(path_fingerprint, exist_ok=True)
    # Load the input image as gray
    input_img = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)
    # Load the Ground Truths and create the "Ground Truth image"
    annotations = utils.GroundTruth.load_annotations(f"{fingerprints_path}/{fingerprint_name}.xml")
    im_squares = utils.GroundTruth.draw_squares(input_img, annotations)
    # Save the ground truth hand-labeled rectangles as sparse images for future template matching
    gt_templates_path = f"{path_fingerprint}/ground_truth_templates/"
    os.makedirs(gt_templates_path, exist_ok=True)
    utils.GroundTruth.extract_template(input_img, annotations, gt_templates_path)
    # Create the pipeline passing the input image
    pipeline = utils.Pipeline(input_img)
    # Schedule the operations to execute
    pipeline.schedule([
        ("Normalization", utils.NormalizationUtils.normalize, [100., 100.]),
        ("Mask", utils.NormalizationUtils.get_fingerprint_mask, [block_size, .2]),
        ("Segmentation", ("Mask", "Normalization"), lambda x, y: x * y, []),
        ("Std-Norm", ("Normalization",), utils.NormalizationUtils.std_norm, []),
        ("Angles", utils.OrientationUtils.calculate_angles, [block_size, ]),
        ("Orientation", ("Mask", "Angles"), utils.OrientationUtils.calculate_orientation, [block_size, ]),
        ("Gabor", ("Std-Norm", "Angles", "Mask"), utils.GaborFilter.apply, [0.65, 0.65, 0.11]),
        ("Skeleton", utils.SkeletonUtils.skeletonize, [], False),
        ("Minutiae", ("Skeleton", "Mask"), utils.MinutiaeUtils.calculate_minutiae, [input_img, ]),
    ])
    # Run the pipeline
    pipeline.execute()
    # Save the pipeline stages images
    pipeline.save_results(fingerprint_name, f"{path_fingerprint}", each=True)
    # Create the analyser with ground truth annotations and the dots predicted as
    # result of pipeline execution.
    analyzer = utils.FingerprintAnalyzer(ground_truth=annotations,
                                         minutiae_points=pipeline.results['Minutiae'][
                                             utils.MinutiaeUtils.dots_filtered])
    # Run the analysis.
    full_matches, matches, misses = analyzer.run_analysis(max_distance=max_distance)
    # Print the matches over the input image and over the skeletonized image.
    im_matches_on_real = utils.GroundTruth.draw_results(input_img, matches, misses)
    im_matches_on_thin = utils.GroundTruth.draw_results(pipeline.results['Skeleton'], matches, misses)
    # Print the full matches over the input image and over the skeletonized image.
    im_full_matches_on_real = utils.GroundTruth.draw_results(input_img, full_matches, misses, full=True)
    im_full_matches_on_thin = utils.GroundTruth.draw_results(pipeline.results['Skeleton'],
                                                             full_matches, misses, full=True)
    # Store the images calculated before
    cv2.imwrite(f"{path_fingerprint}/ground_truth_annotations.png", im_squares)
    cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_on_real.png", im_matches_on_real)
    cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_on_thin.png", im_matches_on_thin)
    cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_full_on_real.png", im_full_matches_on_real)
    cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_full_on_thin.png", im_full_matches_on_thin)
    # Write the matches and misses into csv
    csv_writer_matches.writerows([[fingerprint_name, p[0], p[1], d, gt[0], gt[1], k]for p, d, gt, k in matches])
    csv_writer_misses.writerows([[fingerprint_name, p[0], p[1], k] for p, k in misses])
    csv_writer_full_matches.writerows([[fingerprint_name, p[0], p[1], d, gt[0], gt[1], k]
                                       for p, d, gt, k in full_matches])
    break

csv_file_matches.close()
csv_file_full_matches.close()
csv_file_misses.close()
