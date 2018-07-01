"""
sorting.py

sort the exported images by xy exported from NIS element

Run command:
	python sorting.py

@author: Hsieh-Fu Tsai


"""

import os, sys
import os.path
import shutil

folder_path = "D:\\Data\\20180518_3T3_T98G_300mvmm"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for image in images:
	file_name = image.split('.')[0] #find the file name without the subname
	time_number = file_name.split('t')[-1]
	prexy_name = file_name.split('t')[-2]
	xy_name = prexy_name.split('xy')[-1]
	new_path = os.path.join(folder_path, xy_name)
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	old_image_path = os.path.join(folder_path, image)
	new_image_path = os.path.join(new_path, image)
	shutil.move(old_image_path, new_image_path)

print('organization complete')