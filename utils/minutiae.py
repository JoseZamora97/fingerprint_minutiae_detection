import json

import cv2
import numpy as np
import xmltodict


class MinutiaeUtils:
    @staticmethod
    def _minutiae_at(pixels, i, j, kernel_size):
        # if middle pixel is black (represents ridge)
        if pixels[i][j] == 1:

            if kernel_size == 3:
                cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
                         (0, 1), (1, 1), (1, 0),  # p8    p4
                         (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5
            else:
                cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),  # p1 p2 p3
                         (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),  # p8    p4
                         (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]  # p7 p6 p5

            values = [pixels[i + l][j + k] for k, l in cells]

            # count crossing how many times it goes from 0 to 1
            crossings = 0
            for k in range(0, len(values) - 1):
                crossings += abs(values[k] - values[k + 1])
            crossings //= 2

            # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
            # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
            if crossings == 1:
                return GroundTruth.annotation_termination
            if crossings == 3:
                return GroundTruth.annotation_bifurcation

        return None

    minutiae_image = "image"
    dots = "dots"

    @staticmethod
    def calculate_minutiae(im, kernel_size=3):
        binary_image = np.zeros_like(im)
        binary_image[im < 10] = 1.0
        binary_image = binary_image.astype(np.int8)

        (y, x) = im.shape
        result = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB)
        colors = {GroundTruth.annotation_termination: (150, 0, 0),
                  GroundTruth.annotation_bifurcation: (0, 150, 0)}

        minutiae = []
        for i in range(1, x - kernel_size // 2):
            for j in range(1, y - kernel_size // 2):
                minutiae_kind = MinutiaeUtils._minutiae_at(binary_image, j, i, kernel_size)
                if minutiae_kind:
                    cv2.circle(result, (i, j), radius=2, color=colors[minutiae_kind], thickness=2)
                    minutiae.append(((i, j), minutiae_kind))

        return {MinutiaeUtils.minutiae_image: result, MinutiaeUtils.dots: minutiae}


class GroundTruth:
    kind = 'kind'
    box = 'box'

    annotation_bifurcation = 'bifurcation'
    annotation_termination = 'termination'

    colors = {
        annotation_termination: (255, 0, 0),
        annotation_bifurcation: (0, 255, 0)
    }

    minutiae_colors = {
        annotation_termination: (214, 165, 0),
        annotation_bifurcation: (5, 94, 0)
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
                          cls.colors[gt[GroundTruth.kind]], 1)

        return img_out

    @classmethod
    def draw_matches(cls, img: np.array, matches: list, full=False):
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if full:
            for point, distance, center in matches:
                cv2.line(result, point, center, (0, 68, 140), 1)

                cv2.circle(result, point, 8, (0, 68, 140), 1)
                cv2.circle(result, center, 8, (255, 0, 0), 1)
        else:
            for point, distance, center, kind in matches:
                cv2.line(result, point, center, (0, 68, 140), 1)

                cv2.circle(result, point, 8, GroundTruth.colors[kind], 1)
                cv2.circle(result, center, 8, GroundTruth.minutiae_colors[kind], 1)

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
