import os
from math import sqrt

import cv2
import matplotlib.pyplot as plt

import utils


class FingerprintAnalyzer:
    header_matches = ("point_x", "point_y", "distance", "ground_truth_x", "ground_truth_y", "kind")
    header_full_matches = ("point_x", "point_y", "distance", "ground_truth_x", "ground_truth_y", "kind")
    header_misses = ("point_x", "point_y", "kind")

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

        full_matches, matches, misses = [], [], set()
        for center, kind in centers:
            for point, minutiae_kind in self.minutiae_points:
                distance = self.in_circle(center, point)
                if distance <= max_distance:
                    if kind == minutiae_kind:
                        matches.append((point, distance, center, minutiae_kind))
                    full_matches.append((point, distance, center, minutiae_kind))
                else:
                    misses.add((point, minutiae_kind))

        misses = list(filter(lambda x: x[0] not in list(map(lambda x: x[0], full_matches)), misses))

        return full_matches, matches, misses


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
        def __init__(self, function, args):
            self.function = function
            self.args = args

        def execute(self):
            return self.function(*self.args)

    def __init__(self, im):
        self.tasks: list = list()

        self.last_result = 'Original'
        self.results = {self.last_result: im, }

    def schedule(self, tasks: list):
        self.tasks = tasks

    def execute(self):
        while len(self.tasks) > 0:
            task = self.tasks.pop(0)
            if type(task[0]) is str:
                if type(task[1]) is tuple:
                    self.results[task[0]] = Pipeline.Task(task[2],
                                                          args=[self.results[p]
                                                                for p in task[1]] + task[3],
                                                          ).execute()
                    self.last_result = task[0]

                elif callable(task[1]):
                    self.results[task[0]] = Pipeline.Task(task[1],
                                                          args=[self.results[self.last_result]] + task[2],
                                                          ).execute()
                    self.last_result = task[0]

    def save_results(self, img_name: str, output: str, each=True):
        images_n_titles = list()

        if each:
            os.makedirs(f"{output}/stages", exist_ok=True)

        for title, data in self.results.items():
            if title in ('Angles', 'Original'):
                continue

            if title in ('Mask', 'Std-Norm'):
                im = 255 * data
            else:
                if isinstance(data, dict):
                    im = data[utils.MinutiaeUtils.minutiae_image_filtered]
                else:
                    im = data

            if each:
                cv2.imwrite(f"{output}/stages/{title}.png", im)

            images_n_titles.append((im, title))

        plot(images_n_titles, 4, 2, f'Results - {img_name}', 'gray',
             output=f"{output}/pipeline_{img_name}.png")
