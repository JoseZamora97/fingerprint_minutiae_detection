import os
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob
from math import sqrt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from tqdm import tqdm

import utils


class GroundTruth:

    kind = 'kind'
    box = 'box'

    annotation_bifurcation = 'bifurcation'
    annotation_termination = 'termination'

    colors = {
        annotation_termination: (255, 0, 0),
        annotation_bifurcation: (0, 255, 0)
    }

    @staticmethod
    def load_annotations(xml_path):
        with open(xml_path) as xml_annotations:
            data = xmltodict.parse(xml_annotations.read())
        return [
            {GroundTruth.kind: value['name'], GroundTruth.box: value['bndbox']}
            for value in data['annotation']['object']
        ]

    @classmethod
    def draw_squares(cls, img: np.array, ground_trues: list):
        for gt in ground_trues:
            x0, y0, x, y = gt[GroundTruth.box].values()
            cv2.rectangle(img, (int(x0), int(y0)), (int(x), int(y)),
                          cls.colors[gt[GroundTruth.kind]], 2)

        return img

    @classmethod
    def draw_matches(cls, img: np.array, matches: list):
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        minutiae_colors = {'termination': (214, 165, 0), 'bifurcation': (5, 94, 0)}
        colors = {'termination': (255, 0, 0), 'bifurcation': (0, 255, 0)}

        for point, distance, center, kind in matches:
            cv2.line(result, point, center, (0, 68, 140), 1)
            cv2.circle(result, point, 5, colors[kind], 1)
            cv2.circle(result, center, 5, minutiae_colors[kind], 1)

        return result


class FingerprintAnalyzer:

    def __init__(self, ground_truth, minutiae_points):
        self.ground_truth = ground_truth
        self.minutiae_points = minutiae_points

    @classmethod
    def in_circle(cls, center: tuple, point: tuple):
        distance = sqrt(((point[1] - center[1]) ** 2) + ((point[0] - center[0]) ** 2))
        return distance

    @classmethod
    def calculate_center(cls, a0, b0, a, b):
        return int((int(a0) + int(a)) / 2), int((int(b0) + int(b)) / 2)

    def run_analysis(self, max_distance):
        centers = []

        for ground_truth in self.ground_truth:
            x0, y0, x, y = ground_truth[GroundTruth.box].values()
            centers.append((self.calculate_center(x0, y0, x, y), ground_truth[GroundTruth.kind]))

        matches = []
        for center, kind in centers:
            for point, minutiae_kind in self.minutiae_points:
                if kind == minutiae_kind:
                    distance = self.in_circle(center, point)
                    if distance <= max_distance:
                        matches.append((point, distance, center, minutiae_kind))

        return matches


class Pipeline:

    def __init__(self, block_size=16, workers=12):
        self.block_size = block_size
        self.job_executor = ThreadPoolExecutor(max_workers=workers)


def pipeline(input_img: np.array):
    block_size = 16

    im_normalized = utils.NormalizationUtils.normalize_img(input_img.copy(), float(100), float(100))

    mask = utils.NormalizationUtils.get_fingerprint_mask(im_normalized, block_size, 0.2)
    im_segmented = mask * im_normalized
    im_norm = utils.NormalizationUtils.std_norm(im_normalized)

    angles = utils.OrientationUtils.calculate_angles(im_norm, W=block_size)
    im_orientation = utils.OrientationUtils.visualize_angles(mask, angles, W=block_size)

    im_gabor = utils.GaborFilter.apply(im_norm, angles, mask)
    im_skeleton = utils.SkeletonUtils.skeletonize(im_gabor)
    minutiae, minutiae_points = utils.MinutiaeUtils.calculate_minutiae(im_skeleton)

    return dict(im_normalized=im_normalized, im_mask=mask,
                im_segmented=im_segmented, im_norm=im_norm,
                im_orientation=im_orientation, im_gabor=im_gabor,
                im_thin_image=im_skeleton, im_minutiae=minutiae,
                minutiae_points=minutiae_points)


def process(file_path: str):
    input_img = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
    filename, _ = os.path.splitext(file_path)

    pipeline_result = pipeline(input_img)
    gt_annotations = GroundTruth.load_annotations(f'{filename}.xml')

    analyzer = FingerprintAnalyzer(ground_truth=gt_annotations,
                                   minutiae_points=pipeline_result['minutiae_points'])

    matches = analyzer.run_analysis(max_distance=20)
    im_with_matches = GroundTruth.draw_matches(input_img, matches)

    return im_with_matches, matches, pipeline_result


def plot(ims_n_titles: list, cols: int, rows: int, title: str, cmap: str, f: tuple = (3.5, 4), output: str = None):
    fig = plt.figure(figsize=(cols * f[0], rows * f[1]))
    fig.suptitle(title, fontsize=16)

    i = 1
    for e, t in ims_n_titles:
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(e, cmap=cmap)
        ax.set_title(t)
        ax.axis('off')
        i += 1

    plt.tight_layout()
    if output:
        plt.savefig(output)
        plt.close()
    else:
        plt.show()


def save_results(results_to_save: dict, img_name: str, output: str):
    plot([(v, k) for k, v in results_to_save.items()], 4, 2, f'Results - {img_name}', 'gray', output=output)


if __name__ == '__main__':
    fingerprints_path = './fingerprints/*.tif'
    results_path = './results_to_save.csv'
    output_path = './fingerprints_results'
    os.makedirs(output_path, exist_ok=True)

    hits_dict = {}
    for filepath in tqdm(glob(fingerprints_path)):
        im_hits, hits, results = process(filepath)

        hits_dict[filepath] = (im_hits, hits)
        del results['minutiae_points']
        save_results(results, os.path.basename(filepath),
                     output_path + '/' + os.path.splitext(os.path.basename(filepath))[0] + '.png')

    # TODO: fix: pd.DataFrame.from_dict(hits_dict).to_csv(results_path)
