"""
Mask-RCNN training script.

Usage:
    train.py --train-dir=<dataset_dir> --val-dir=<val_dir>
             --out-dir=<out_dir> [--weights=<weights_path>]
"""
import os

import cv2

import numpy as np
from docopt import docopt
from imgaug import augmenters as iaa

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config

import sys

INPUT_CHANNEL = 'raw.tif'


class cellConfig(Config):
    NAME = "cell"

    # Adjust to GPU memory
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2  # Background + cell

    STEPS_PER_EPOCH = 45
    VALIDATION_STEPS = 5

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5
    #DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    #BACKBONE = "resnet50"
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    #RPN_NMS_THRESHOLD = 0.9
    RPN_NMS_THRESHOLD=0.99

    # How many anchors per image to use for RPN training
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([126,126,126])
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    MINI_MASK_SHAPE = (100,100)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    #TRAIN_ROIS_PER_IMAGE = 128
    TRAIN_ROIS_PER_IMAGE = 256

    # Maximum number of ground truth instances to use in one image
    #MAX_GT_INSTANCES = 200
    MAX_GT_INSTANCES = 500
    # Max number of final detections per image
    #DETECTION_MAX_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 1000

class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open("log.log", "a")
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
	def flush(self):
		pass


############################################################
#  Dataset
############################################################


class cellDataset(utils.Dataset):

    def load_cell(self, dataset_dir):
        """
        Load dataset
        :param dataset_dir: Root directory to dataset
        :return:
        """
        self.add_class("cell", 1, "cell")

        image_ids = os.listdir(dataset_dir)

        for image_id in image_ids:
            self.add_image(
                'cell',
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, INPUT_CHANNEL)
            )

        print('[DATASET]', dataset_dir, len(self.image_ids))

    def load_mask(self, image_id):
        """

        :param image_id:
        :return:
        """
        info = self.image_info[image_id]

        mask_dir = os.path.dirname(info['path'])
        mask_path = os.path.join(mask_dir, 'instances_ids.png')

        ids_mask = cv2.imread(mask_path, 0)
        instances_num = len(np.unique(ids_mask)) - 1

        mask = np.zeros((ids_mask.shape[0], ids_mask.shape[1], instances_num))
        for i in range(instances_num):
            # print(np.where(ids_mask == (i + 1)))
            slice = mask[..., i]
            slice[np.where(ids_mask == (i + 1))] = 1
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """

        :param image_id:
        :return:
        """
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, config, train_dir, val_dir):
    train_dataset = cellDataset()
    train_dataset.load_cell(train_dir)
    train_dataset.prepare()

    val_dataset = cellDataset()
    val_dataset.load_cell(val_dir)
    val_dataset.prepare()

    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
        #iaa.GaussianBlur(sigma=(0,3.0))
    ])

    print("Train network heads")
    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE,
                #learning_rate = 0.01,
                epochs=100,
                augmentation=augmentation,
                layers='heads')
    
    print("Train all layers")
    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE,
                #learning_rate = 0.1,
                epochs=200,
                augmentation=augmentation,
                layers='all')



def main():
    # args = docopt(__doc__)
    #
    # train_dir = args['--train-dir']
    # val_dir = args['--val-dir']
    # out_dir = args['--out-dir']
    # weights_path = args['--weights']
    sys.stdout = Logger()
    #define the path to the train directory containing instance_ids.png and raw.tif
    train_dir = "/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/train/"
    #define the path to the validation directory containing instance_ids.png and raw.tif
    val_dir = "/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/val/"
    #define the path to the directory that output trained weights should be saved
    out_dir = "/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/trainednetwork/"
    #define the path of pretrained weight to start from
    weights_path = '/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/trainednetwork/mask_rcnn_nuclei_new.h5'

    print('> Training on data: ', train_dir)
    print('> Saving results to: ', out_dir)

    config = cellConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=out_dir)

    if weights_path is not None:
        print('> Loading weights from: ', weights_path)
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"
        ])

    train(model, config, train_dir, val_dir)


if __name__ == '__main__':
    main()
