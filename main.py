import csv
import os
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob
from math import sqrt

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


class FingerprintAnalyzer:

    header_matches = ("point", "distance", "ground_truth", "kind")
    header_full_matches = ("point", "distance", "ground_truth")

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
            x0, y0, x, y = ground_truth[utils.GroundTruth.box].values()
            centers.append((self.calculate_center(x0, y0, x, y),
                            ground_truth[utils.GroundTruth.kind]))

        full_matches, matches = [], []
        for center, kind in centers:
            for point, minutiae_kind in self.minutiae_points:
                distance = self.in_circle(center, point)
                if distance <= max_distance:
                    if kind == minutiae_kind:
                        matches.append((point, distance, center, minutiae_kind))
                    full_matches.append((point, distance, center))

        return full_matches, matches


def plot(ims_n_titles: list, cols: int, rows: int, title: str, cmap: str, output: str = None):
    fig = plt.figure()
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


class Pipeline:
    class Task:
        def __init__(self, function, args, executor):
            self.function = function
            self.args = args
            self.executor = executor

        def execute(self):
            if self.executor:
                return self.function(*self.args, executor=self.executor)
            else:
                return self.function(*self.args)

    def __init__(self, input_img, workers=12):
        self.job_executor = ThreadPoolExecutor(max_workers=workers)
        self.tasks: list = list()

        self.last_result = 'Original'
        self.results = {self.last_result: input_img, }

    def schedule(self, tasks: list):
        self.tasks = tasks

    def execute(self):
        while len(self.tasks) > 0:
            task = self.tasks.pop(0)
            if type(task[0]) is str:
                if type(task[1]) is tuple:
                    self.results[task[0]] = Pipeline.Task(task[2], args=[self.results[p] for p in task[1]] + task[3],
                                                          executor=self.job_executor if task[4] else None).execute()
                    self.last_result = task[0]

                elif callable(task[1]):
                    self.results[task[0]] = Pipeline.Task(task[1], args=[self.results[self.last_result]] + task[2],
                                                          executor=self.job_executor if task[3] else None).execute()
                    self.last_result = task[0]

    @classmethod
    def save_results(cls, results_to_save: dict, img_name: str, output: str, each=True):
        images_n_titles = list()

        if each:
            os.makedirs(f"{output}/stages", exist_ok=True)

        for title, data in results_to_save.items():
            if title in ('Angles', 'Original'):
                continue

            if title in ('Mask', 'Std-Norm'):
                im = 255 * data

            else:
                if isinstance(data, dict):
                    im = data['image']
                else:
                    im = data

            if each:
                cv2.imwrite(f"{output}/stages/{title}.png", im)
            images_n_titles.append((im, title))

        plot(images_n_titles, 4, 2, f'Results - {img_name}', 'gray',
             output=f"{output}/pipeline_{img_name}.png")


def process(input_img, gt_annotations, block_size=16, max_distance=20):
    pipeline = Pipeline(input_img, workers=50)
    pipeline.schedule([
        ("Normalization", utils.NormalizationUtils.normalize, [100., 100.], False),
        ("Mask", utils.NormalizationUtils.get_fingerprint_mask, [block_size, .2], False),
        ("Segmentation", ("Mask", "Normalization"), lambda x, y: x * y, [], False),
        ("Std-Norm", ("Normalization",), utils.NormalizationUtils.std_norm, [], False),
        ("Angles", utils.OrientationUtils.calculate_angles, [block_size, ], False),
        ("Orientation", ("Mask", "Angles"), utils.OrientationUtils.calculate_orientation, [block_size, ], False),
        ("Gabor", ("Std-Norm", "Angles", "Mask"), utils.GaborFilter.apply, [0.65, 0.65, 0.11], False),
        ("Skeleton", utils.SkeletonUtils.skeletonize, [], False),
        ("Minutiae", utils.MinutiaeUtils.calculate_minutiae, [], False),
    ])

    pipeline.execute()

    analyzer = FingerprintAnalyzer(ground_truth=gt_annotations,
                                   minutiae_points=pipeline.results['Minutiae'][utils.MinutiaeUtils.dots])

    full_matches, matches = analyzer.run_analysis(max_distance=max_distance)

    return full_matches, matches, pipeline.results


if __name__ == '__main__':

    fingerprints_path = './fingerprints'
    output_path = './fingerprints_results_pipeline'

    os.makedirs(output_path, exist_ok=True)

    csv_file_matches = open(f"{output_path}/results_matches.csv", "w")
    csv_file_full_matches = open(f"{output_path}/results_full_matches.csv", "w")

    csv_writer_matches = csv.writer(csv_file_matches)
    csv_writer_matches.writerow(["file"] + [e for e in FingerprintAnalyzer.header_matches])

    csv_writer_full_matches = csv.writer(csv_file_full_matches)
    csv_writer_full_matches.writerow(["file"] + [e for e in FingerprintAnalyzer.header_full_matches])

    for filepath in tqdm(glob(f"{fingerprints_path}/*.tif"), desc='Analysing Fingerprints', position=0):

        fingerprint_name = os.path.splitext(os.path.basename(filepath))[0]
        path_fingerprint = f"{output_path}/{fingerprint_name}"
        os.makedirs(path_fingerprint, exist_ok=True)

        input_img = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)

        annotations = utils.GroundTruth.load_annotations(f"{fingerprints_path}/{fingerprint_name}.xml")
        im_squares = utils.GroundTruth.draw_squares(input_img, annotations)

        full_matches, matches, results = process(input_img, annotations, block_size=16, max_distance=8)

        im_matches_on_real = utils.GroundTruth.draw_matches(input_img, matches)
        im_matches_on_thin = utils.GroundTruth.draw_matches(results['Skeleton'], matches)

        im_full_matches_on_real = utils.GroundTruth.draw_matches(input_img, full_matches, full=True)
        im_full_matches_on_thin = utils.GroundTruth.draw_matches(results['Skeleton'], full_matches, full=True)

        Pipeline.save_results(results, fingerprint_name, f"{path_fingerprint}", each=True)

        gt_templates_path = f"{path_fingerprint}/ground_truth_templates/"
        os.makedirs(gt_templates_path, exist_ok=True)
        utils.GroundTruth.extract_template(input_img, annotations, gt_templates_path)

        cv2.imwrite(f"{path_fingerprint}/ground_truth_annotations.png", im_squares)
        cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_on_real.png", im_matches_on_real)
        cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_on_thin.png", im_matches_on_thin)
        cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_full_on_real.png", im_full_matches_on_real)
        cv2.imwrite(f"{path_fingerprint}/matches_{fingerprint_name}_full_on_thin.png", im_full_matches_on_thin)

        for m in matches:
            csv_writer_matches.writerow([fingerprint_name, ] + [u for u in m])

        for m in full_matches:
            csv_writer_full_matches.writerow([fingerprint_name, ] + [u for u in m])

    csv_file_matches.close()
    csv_file_full_matches.close()
