"""
Mask-RCNN detection script for cell

Input format
> data_dir: Directory with structure as follows:
    - <data_dir>
        - <image_id_1.tif>   (file name is it's id)
        - <image_id_2.tif>
        - <image_id_3.tif>

> output_dir: Directory to save the predictions, format after prediction:
    - <output_dir>
        - <image_id_1_mask.png>
        - <image_id_2_mask.png>

> model_path: Path to hdf5 file with saved model.

added with postprocessing on sequential predictions from multiple models
supported cases:
- cell notdetected by one of the models
- cell not deted in one of few in the sequence
- single cell detected as two cells


"""
import sys
import datetime
import os

import numpy as np
import cv2
from tqdm import tqdm

from train import cellConfig
from mrcnn import utils
from mrcnn import model as modellib
import time

from itertools import groupby
from operator import itemgetter

import operator


IOU_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.8
MIN_DETECTIONS = 1

"""
logging class
"""


class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open(data_dir+"log.log", "a")
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
	def flush(self):
		pass


class ImageDataset(utils.Dataset):
    def load_images(self, dataset_dir):
        """
        Loads dataset images.
        :param dataset_dir: string, path to dataset directory.
        :return: None
        """
        self.add_class("cell", 1, "cell")

        image_ids = os.listdir(dataset_dir)

        for image_id in image_ids:
            self.add_image(
                'cell',
                image_id=os.path.splitext(image_id)[0],
                path=os.path.join(dataset_dir, image_id)
            )


class CellInferenceConfig(cellConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


def detect(model, data_dir, out_dir):
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    detection_dir = "detections_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    detection_dir = os.path.join(out_dir, detection_dir)
    os.makedirs(detection_dir)
	'''
    # Read dataset
    dataset = ImageDataset()
    dataset.load_images(data_dir)
    dataset.prepare()
    # Load over images
    for image_id in tqdm(dataset.image_ids):
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]

        #out_path = os.path.join(detection_dir, '%s.png' % str(source_id))
        out_path = os.path.join(out_dir, '%s.png' % str(source_id))

        mask = np.argmax(r['masks'], 2)
        cv2.imwrite(out_path, mask)



def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union score for two binary masks.
    :param mask1: numpy array
    :param mask2: numpy array
    :return:
    """
    intersection = np.sum((mask1 + mask2) > 1)
    union = np.sum((mask1 + mask2) > 0)

    return intersection / float(union)


def compute_overlap(mask1, mask2):
    intersection = np.sum((mask1 + mask2) > 1)

    overlap1 = intersection / float(np.sum(mask1))
    overlap2 = intersection / float(np.sum(mask2))
    return overlap1, overlap2


def sort_mask_by_cells(mask, min_size=50):
    """
    Returns size of each cell.
    :param mask:
    :return:
    """
    cell_num = np.unique(mask)
    cell_sizes = [(cell_id, len(np.where(mask == cell_id)[0])) for cell_id in cell_num if cell_id != 0]

    cell_sizes = [x for x in sorted(cell_sizes, key=lambda x: x[1], reverse=True) if x[1 > min_size]]

    return cell_sizes


def merge_multiple_detections(masks):
    """

    :param masks:
    :return:
    """
    cell_counter = 0
    final_mask = np.zeros(masks[0].shape)

    masks_stats = [sort_mask_by_cells(mask) for mask in masks]
    cells_left = sum([len(stats) for stats in masks_stats])

    while cells_left > 0:
        # Choose the biggest cell from available
        cells = [stats[0][1] if len(stats) > 0 else 0 for stats in masks_stats]
        reference_mask = cells.index(max(cells))

        reference_cell = masks_stats[reference_mask].pop(0)[0]

        # Prepare binary mask for cell chosen for comparison
        cell_location = np.where(masks[reference_mask] == reference_cell)

        cell_mask = np.zeros(final_mask.shape)
        cell_mask[cell_location] = 1

        masks[reference_mask][cell_location] = 0

        # Mask for storing temporary results
        tmp_mask = np.zeros(final_mask.shape)
        tmp_mask += cell_mask

        for mask_id, mask in enumerate(masks):
            # For each mask left
            if mask_id != reference_mask:
                # # Find overlapping cells on other masks
                overlapping_cells = list(np.unique(mask[cell_location]))

                try:
                    overlapping_cells.remove(0)
                except ValueError:
                    pass

                # # If only one overlapping, check IoU and update tmp mask if high
                if len(overlapping_cells) == 1:
                    overlapping_cell_mask = np.zeros(final_mask.shape)
                    overlapping_cell_mask[np.where(mask == overlapping_cells[0])] = 1

                    iou = compute_iou(cell_mask, overlapping_cell_mask)
                    if iou >= IOU_THRESHOLD:
                        # Add cell to temporary results and remove from stats and mask
                        tmp_mask += overlapping_cell_mask
                        idx = [i for i, cell in enumerate(masks_stats[mask_id]) if cell[0] == overlapping_cells[0]][0]
                        masks_stats[mask_id].pop(idx)
                        mask[np.where(mask == overlapping_cells[0])] = 0

                # # If more than one overlapping check area overlapping
                elif len(overlapping_cells) > 1:
                    overlapping_cell_masks = [np.zeros(final_mask.shape) for _ in overlapping_cells]

                    for i, cell_id in enumerate(overlapping_cells):
                        overlapping_cell_masks[i][np.where(mask == cell_id)] = 1

                    for cell_id, overlap_mask in zip(overlapping_cells, overlapping_cell_masks):
                        overlap_score, _ = compute_overlap(overlap_mask, cell_mask)

                        if overlap_score >= OVERLAP_THRESHOLD:
                            tmp_mask += overlap_mask

                            mask[np.where(mask == cell_id)] = 0
                            idx = [i for i, cell in enumerate(masks_stats[mask_id])
                                   if cell[0] == cell_id][0]
                            masks_stats[mask_id].pop(idx)

                # # If none overlapping do nothing

        if len(np.unique(tmp_mask)) > 1:
            cell_counter += 1
            final_mask[np.where(tmp_mask >= MIN_DETECTIONS)] = cell_counter

        cells_left = sum([len(stats) for stats in masks_stats])

    bin_mask = np.zeros(final_mask.shape)
    bin_mask[np.where(final_mask > 0)] = 255

    cv2.imwrite('results/final_bin.png', bin_mask)
    cv2.imwrite('results/final.png', final_mask)
    return final_mask


def process_sequence(masks):
    """

    :param masks:
    :return:
    """


def postprocess(data_dir, out_dir):
    """

    :param data_dir:
    :param out_dir:
    :return:
    """
    #os.makedirs(out_dir, exist_ok=True)
    models_dir = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    print('Merging multiple models predictions.')
    filenames = os.listdir(models_dir[0])

    for filename in tqdm(filenames):
        masks = [cv2.imread(os.path.join(model_dir, filename), 0) for model_dir in models_dir]

        result = merge_multiple_detections(masks)
        bin_result = np.zeros(result.shape)
        bin_result[np.where(result > 0)] = 255

        cv2.imwrite(os.path.join(out_dir, filename), result)
        #cv2.imwrite(os.path.join(out_dir, 'bin_%s' % filename), bin_result)
        #cv2.imwrite(os.path.join(out_dir, filename), bin_result)

    # print('Processing sequence.')
    # masks = [cv2.imread(os.path.join(out_dir, filename), 0) for filename in filenames]
    # results = process_sequence(masks)
    #
    # for filename, result in zip(filenames, results):
    #     cv2.imwrite(os.path.join(out_dir, filename), result)


#define the folder path to data for prediction
data_dir = '/media/davince/DATA_HD/Cell electrotaxis/20180603similarity_analysis/3T3Ctrl/' #don't forget the / at the end
#model_path = '/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/trainednetwork/mask_rcnn_nuclei_res101.h5'

#define the model weight paths
model_path_1 = '/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/trainednetwork/trained/mask_rcnn_nuclei_0521_0.2089.h5'
model_path_2 = '/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/trainednetwork/trained/mask_rcnn_nuclei_res101.h5'
model_path_3 = '/home/davince/Dropbox (OIST)/Deeplearning_system/Mask-RCNN_OIST/trainednetwork/mask_rcnn_nuclei_0750.h5'
#model_list=[model_path_3]
model_list = [model_path_1, model_path_2, model_path_3]
#model_list = [model_path_1, model_path_2]
# find all the subdirectory by first identifying all the directory in all levels, then add to a sub_directory list
#the intended prediction files should be organized by experiments into sets in a main folder which will be the direc_name
all_files = []
sub_directory = []
for root, dirs, files in os.walk(data_dir):
	for file in files:
		relativePath = os.path.relpath(root, data_dir)
		if relativePath == ".":
			relativePath = ""
		all_files.append((relativePath.count(os.path.sep),relativePath, file))
all_files.sort(reverse=True)
for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
	sub_directory.append(folder)
#print(sub_directory)

config = CellInferenceConfig()
#model = modellib.MaskRCNN(mode="inference", config=config, model_dir=out_dir)
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=data_dir)
#print('> Loading model from: ', model_path)
#model.load_weights(model_path, by_name=True)
sys.stdout = Logger()

#iterate through each subfolder

#change model weights with each directory
total_start = time.time()
run_time_log = []
for i in sub_directory:
	counter = 1
	start = time.time()
	mask_duplicate_dir = os.path.join(data_dir, i+'_mask')
	for m in model_list:
		#print(m)
		print('>Loading model from: ', m)
		model.load_weights(m, by_name=True)
		try:
			predict_location = os.path.join(data_dir, i)
			print('prediction for: ', predict_location)
			print('model run '+str(counter)+' of '+str(len(model_list)))
			try:
				out_dir= os.path.join(mask_duplicate_dir, "_"+str(counter))
				os.makedirs(out_dir)
			except:
				print('failed to create mask folder')
			#print('out_dir is'+out_dir)
			try:
				detect(model, predict_location, out_dir)
			except:
				print('failed to deploy the inference, skipping...')
		except:
			print('error, skipping...')
		counter += 1
	try:
		avg_prediction_dir = os.path.join(data_dir, i+"_mask_avg")
		os.makedirs(avg_prediction_dir)
	except:
		print('failed to create avg mask folder')
	postprocess(mask_duplicate_dir, avg_prediction_dir)
	end = time.time()
	time_diff = end-start
	run_time_log.append(time_diff)
	hour = time_diff // 3600
	time_diff %= 3600
	minutes = time_diff // 60
	time_diff %= 60
	seconds = time_diff
	print('prediction run time = %d hr: %d min: %d s'%(hour, minutes, seconds))
print(run_time_log)
runtimelogfile=open('exptime.txt', 'w')
for item in run_time_log:
    runtimelogfile.write("%s\n" % item)
total_end = time.time()
total_time = total_end - total_start
total_day = total_time // (3600*24)
total_time %= (3600*24)
total_hour = total_time //3600
total_time %= 3600
total_minutes =total_time //60
total_time %= 60
total_seconds = total_time 
print('Total prediction run time = %d day: %d hr: %d min: %d s'%(total_day, total_hour, total_minutes, total_seconds))
