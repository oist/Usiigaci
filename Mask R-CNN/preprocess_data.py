"""
Script that modifies segmentation mask with instances. Each color
is converted to instance id and image is saved in the dame directory.
Directory structre:
root_dir
    set1
        raw.tif
        labeled.png
        ...
    set2
        raw.tif
        labeled.png
        ...

Usage:
    preprocess_data.py --data-dir=<data_dir>
"""
import os

import cv2
import numpy as np
from docopt import docopt
from tqdm import tqdm


def convert_colors_to_ids(color_mask):
    unique_colors = np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)

    ids_mask = np.zeros(color_mask.shape[:2])

    id_counter = 1
    for color in unique_colors:
        if not (color == [0, 0, 0]).all():
            instance = cv2.inRange(color_mask, color, color)
            ids_mask[np.where(instance > 0)] = id_counter
            id_counter += 1
    return ids_mask


def main():
    # args = docopt(__doc__)

    # data_dir = args['--data-dir']
    data_dir = '/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/val/'

    image_dirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]
    for image_dir in tqdm(image_dirs):
        color_mask = cv2.imread(os.path.join(image_dir, 'labeled.png'))

        ids_mask = convert_colors_to_ids(color_mask)
        cv2.imwrite(os.path.join(image_dir, 'instances_ids.png'), ids_mask)


if __name__ == '__main__':
    main()
    #
    # image = cv2.imread('/data/upwork/cells/instance_aware_seg/data/train/set1/labeled.png')
    # ids = convert_colors_to_ids(image)
    # cv2.imwrite('test.png', ids)
