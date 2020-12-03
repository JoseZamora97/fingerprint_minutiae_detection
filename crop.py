import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_spec(img, t, save=None):
    cols = img.mean(axis=0)
    cols = (cols - np.min(cols)) / (np.max(cols) - np.min(cols))

    rows = img.mean(axis=1)
    rows = (rows - np.min(rows)) / (np.max(rows) - np.min(rows))

    threshold_values = np.array([t for _ in range(cols.size)])
    plt.plot(cols, 'red', rows, 'blue', threshold_values, 'g--')
    plt.legend(['col means', 'row means', 'max_distance'])

    if save:
        plt.savefig(save)

    plt.show()


def get_crop_indexes(array, t):
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


def calculate_crop_indexes(img, t):
    cols = img.mean(axis=0)
    cols = (cols - np.min(cols)) / (np.max(cols) - np.min(cols))
    i_col, j_col = get_crop_indexes(cols, t)

    rows = img.mean(axis=1)
    rows = (rows - np.min(rows)) / (np.max(rows) - np.min(rows))
    i_row, j_row = get_crop_indexes(rows, t)

    return [i_row, j_row, i_col, j_col]


def crop_by_threshold(img, t):
    i_row, j_row, i_col, j_col = calculate_crop_indexes(img, t)
    return img[i_row:j_row, i_col:j_col]


def crop_images(input_directory, output_directory, t):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i in os.listdir(input_directory):
        name, ext = os.path.splitext(i)
        if ext in ['.tif', ]:
            im = 255 - cv2.imread(f"{input_directory}/{i}", cv2.COLOR_BGR2GRAY)
            cropped_im = crop_by_threshold(im, t)
            plot_spec(im, t, f'{output_directory}/{name}.png')
            cv2.imwrite(f'{output_directory}/{i}', cropped_im)


def plot(im, title, cmap='gray'):
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.axis('off'), plt.show()
