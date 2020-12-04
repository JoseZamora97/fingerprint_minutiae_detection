import json

import cv2
import numpy as np
import xmltodict


class Cropper:

    @classmethod
    def get_crop_indexes(cls, array, t):
        i_out = j_out = -1
        for (i, iv), (j, jv) in zip(
                enumerate(array),
                list(enumerate(array))[::-1]
        ):
            if iv >= t and i_out == -1:
                i_out = i
            if jv >= t and j_out == -1:
                j_out = j
            if i_out != -1 and j_out != -1:
                break

        return i_out, j_out

    @classmethod
    def calculate_crop_indexes(cls, img, t):
        cols = img.mean(axis=0)
        cols = (cols - np.min(cols)) / (np.max(cols) - np.min(cols))
        i_col, j_col = cls.get_crop_indexes(cols, t)

        rows = img.mean(axis=1)
        rows = (rows - np.min(rows)) / (np.max(rows) - np.min(rows))
        i_row, j_row = cls.get_crop_indexes(rows, t)

        return [i_row, j_row, i_col, j_col]


class MinutiaeUtils:

    minutiae_image = "image"
    dots = "dots"
    minutiae_image_filtered = "image_filtered"
    dots_filtered = "dots_filtered"

    @classmethod
    def get_bound_indexes(cls, i, j, size):

        bound_top = [(i - size, x) for x in range(j - size, j + size + 1)]
        bound_bottom = [(i + size, x) for x in range(j - size, j + size + 1)]

        bound_left = [(x, j - size) for x in range(i - size, i + size + 1)]
        bound_right = [(x, j + size) for x in range(i - size, i + size + 1)]

        return bound_top, bound_bottom, bound_left, bound_right

    @classmethod
    def _minutiae_at(cls, pixels, i, j, kernel_size, mask):

        kind = kind_f = None

        if pixels[i][j] == 1:

            cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
                     (0, 1), (1, 1), (1, 0),      # p8    p4
                     (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5

            values = [pixels[i + l][j + k] for k, l in cells]
            crossings = 0
            for k in range(0, len(values) - 1):
                crossings += abs(values[k] - values[k + 1])
            crossings //= 2

            if crossings == 1:
                kind = GroundTruth.annotation_termination
                kind_f = GroundTruth.annotation_termination
            elif crossings == 3:
                kind = GroundTruth.annotation_bifurcation
                kind_f = GroundTruth.annotation_bifurcation

            bt, bb, bl, br = cls.get_bound_indexes(i, j, kernel_size * 5)

            btn_m = bbn_m = bln_m = btn_m = 0

            for x in range(len(bt)):
                btn_m += mask[bt[x][0]][bt[x][1]]
                bbn_m += mask[bb[x][0]][bb[x][1]]
                bln_m += mask[bl[x][0]][bl[x][1]]
                btn_m += mask[br[x][0]][br[x][1]]

            if any(map(lambda a: a == 0, [btn_m, bbn_m, bln_m, btn_m])):
                kind_f = None

        return kind, kind_f

    @staticmethod
    def calculate_minutiae(im, mask, original, kernel_size=3):
        binary_image = np.zeros_like(im)
        binary_image[im < 10] = 1.0
        binary_image = binary_image.astype(np.int8)

        result = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB)
        result_filtered = result.copy()

        x0, x, y0, y = Cropper.calculate_crop_indexes(255 - original, t=0.25)

        colors = {GroundTruth.annotation_termination: (150, 0, 0),
                  GroundTruth.annotation_bifurcation: (0, 150, 0)}

        minutiae, minutiae_filtered = [], []

        for i in range(x0, x - kernel_size // 2):
            for j in range(y0, y - kernel_size // 2):

                minutiae_kind, minutiae_kind_filtered = MinutiaeUtils._minutiae_at(
                    binary_image, i, j, kernel_size, mask
                )

                if minutiae_kind:
                    cv2.circle(result, (j, i), radius=2, color=colors[minutiae_kind], thickness=2)
                    minutiae.append(((j, i), minutiae_kind))

                if minutiae_kind_filtered:
                    cv2.circle(result_filtered, (j, i), radius=2, color=colors[minutiae_kind], thickness=2)
                    minutiae_filtered.append(((j, i), minutiae_kind))

        return {
            MinutiaeUtils.dots_filtered: minutiae_filtered,
            MinutiaeUtils.minutiae_image_filtered: result_filtered,

            MinutiaeUtils.minutiae_image: result,
            MinutiaeUtils.dots: minutiae
        }


class GroundTruth:
    kind = 'kind'
    box = 'box'

    annotation_bifurcation = 'bifurcation'
    annotation_termination = 'termination'

    COLORS = {
        "line": {
            annotation_termination: (0, 68, 140),
            annotation_bifurcation: (0, 68, 140)
        },
        "full": {
            annotation_termination: (0, 68, 140),
            annotation_bifurcation: (0, 68, 140)
        },
        "prediction": {
            annotation_termination: (255, 0, 0),
            annotation_bifurcation: (0, 255, 0)
        },
        "errors": {
            annotation_termination: (0, 0, 255),
            annotation_bifurcation: (0, 108, 255)
        },
        "gt": {
            annotation_termination: (214, 165, 0),
            annotation_bifurcation: (5, 94, 0)
        }
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
    def draw_squares(cls, img: np.array, ground_truths: list):
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for gt in ground_truths:
            x0, y0, x, y = gt[GroundTruth.box].values()
            cv2.rectangle(img_out, (int(x), int(y)), (int(x0), int(y0)),
                          cls.COLORS['gt'][gt[GroundTruth.kind]], 1)

        return img_out

    @classmethod
    def draw_results(cls, img: np.array, matches: list, misses, full=False):
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for point, kind in misses:
            cv2.circle(result, point, 1, GroundTruth.COLORS['errors'][kind], 2)

        if full:
            for point, distance, center, kind in matches:
                cv2.line(result, point, center, GroundTruth.COLORS['line'][kind], 1)
                cv2.circle(result, point, 8, GroundTruth.COLORS['full'][kind], 1)
                cv2.circle(result, center, 8, GroundTruth.COLORS['gt'][kind], 1)
        else:
            for point, distance, center, kind in matches:
                cv2.line(result, point, center, GroundTruth.COLORS['line'][kind], 1)
                cv2.circle(result, point, 8, GroundTruth.COLORS['prediction'][kind], 1)
                cv2.circle(result, center, 8, GroundTruth.COLORS['gt'][kind], 1)

        return result

    @classmethod
    def extract_template(cls, input_img, ground_truths, output, size=(20, 20)):
        gt_termination = gt_bifurcation = 0

        for i, gt in enumerate(ground_truths):

            if gt[GroundTruth.kind] == GroundTruth.annotation_bifurcation:
                gt_bifurcation += 1

            if gt[GroundTruth.kind] == GroundTruth.annotation_termination:
                gt_termination += 1

            x0, y0, x, y = gt[GroundTruth.box].values()
            kind = gt[GroundTruth.kind]

            im = input_img[int(y0):int(y), int(x0):int(x)]
            cv2.imwrite(f"{output}/{i}_{kind}.png",
                        cv2.resize(im, size, interpolation=cv2.INTER_LINEAR))

        with open(f"{output}inform.json", "w") as inform:
            inform.write(json.dumps({
                GroundTruth.annotation_termination: gt_termination,
                GroundTruth.annotation_bifurcation: gt_bifurcation
            }, indent=4))
