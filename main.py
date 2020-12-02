import os
from glob import glob
from math import sqrt

import cv2
import pandas as pd
from tqdm import tqdm

import utils
import xmltodict
import numpy as np
import matplotlib.pyplot as plt


def get_ground_true_annotations(xml_path: str):
    with open(xml_path) as xml:
        dict_data = xmltodict.parse(xml.read())
        objects = [{'type': value['name'], 'box': value['bndbox']} for value in dict_data['annotation']['object']]
        return objects


def draw_squares(img: np.array, ground_trues: list):
    colors = {'termination': (255, 0, 0), 'bifurcation': (0, 255, 0)}
    for ground_true in ground_trues:
        x0, y0, x, y = ground_true['box'].values()
        cv2.rectangle(img, (int(x0), int(y0)), (int(x), int(y)), colors[ground_true['type']], 2)
    return img


def pipeline(input_img: np.array):
    block_size = 16

    im_normalized = utils.NormalizationUtils.normalize_img(input_img.copy(), float(100), float(100))

    mask = utils.NormalizationUtils.get_fingerprint_mask(im_normalized, block_size, 0.2)
    im_segmented = mask * im_normalized
    im_norm = utils.NormalizationUtils.std_norm(im_normalized)

    im_angles = utils.OrientationUtils.calculate_angles(im_norm, W=block_size)
    im_orientation = utils.OrientationUtils.visualize_angles(mask, im_angles, W=block_size)

    gabor_img = utils.GaborFilter.apply(im_norm, im_angles, mask)

    thin_image = utils.SkeletonUtils.skeletonize(gabor_img)

    minutiae, minutiae_points = utils.MinutiaeUtils.calculate_minutiae(thin_image)

    return dict(im_normalized=im_normalized, im_mask=mask,
                im_segmented=im_segmented, im_norm=im_norm,
                im_orientation=im_orientation, im_gabor=gabor_img,
                im_thin_image=thin_image, im_minutiae=minutiae,
                minutiae_points=minutiae_points)


def in_circle(center: tuple, point: tuple):
    distance = sqrt(((point[1] - center[1]) ** 2) + ((point[0] - center[0]) ** 2))
    return distance


def analysis(ground_trues: list, minutiae_points: list, threshold: int = 15):
    centers = []
    center = lambda x0, y0, x, y: (int((int(x0) + int(x)) / 2), int((int(y0) + int(y)) / 2))
    for ground_truth in ground_trues:
        x0, y0, x, y = ground_truth['box'].values()
        centers.append((center(x0, y0, x, y), ground_truth['type']))

    hits = []
    for center, type in centers:
        for point, mtype in minutiae_points:
            if type == mtype:
                distance = in_circle(center, point)
                if distance < threshold:
                    hits.append((point, distance, center, mtype))

    return hits


def draw_hits(img: np.array, hits: list):
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mcolors = {'termination': (214, 165, 0), 'bifurcation': (5, 94, 0)}
    colors = {'termination': (255, 0, 0), 'bifurcation': (0, 255, 0)}
    for point, distance, center, type in hits:
        cv2.line(result, point, center, (0, 68, 140), 1)
        cv2.circle(result, point, 5, colors[type], 1)
        cv2.circle(result, center, 5, mcolors[type], 1)
    return result


def process(file_path: str):
    input_img = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)

    results = pipeline(input_img)

    filename, _ = os.path.splitext(file_path)
    ground_trues = get_ground_true_annotations(f'{filename}.xml')

    hits = analysis(ground_trues, results['minutiae_points'])

    im_hits = draw_hits(input_img, hits)

    return im_hits, hits, results


def plot(ims_n_titles: list, cols: int, rows: int, title: str, cmap: str, f: tuple = (3.5, 4), output: str = None):
    fig=plt.figure(figsize=(cols * f[0], rows * f[1]))
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


def save_results(results: dict, img_name: str, output: str):
    plot([(v, k) for k, v in results.items()], 4, 2, f'Results - {img_name}', 'gray', output=output)


if __name__ == '__main__':
    fingerprints_path = './fingerprints/*.tif'
    results_path = './results.csv'
    output_path = './fingerprints_results'
    os.makedirs(output_path, exist_ok=True)

    hits_dict = {}
    for filepath in tqdm(glob(fingerprints_path)):
        im_hits, hits, results = process(filepath)

        hits_dict[filepath] = (im_hits, hits)
        del[results['minutiae_points']]
        save_results(results, os.path.basename(filepath),
                     output_path + '/' + os.path.splitext(os.path.basename(filepath))[0] + '.png')

    pd.DataFrame.from_dict(hits_dict).to_csv(results_path)
