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
#use "C:\\dslds\\sdfd\\" on windows
folder_path = "D:\\Cell electrotaxis\\20180603similarity_analysis\\3T3Ctrl\\02"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for image in images:
	file_name = image.split('.')[0] #find the file name without the subname
	if file_name.split('c')[-1].isdigit() is True:
		channel_name = file_name.split('c')[-1]
		time_number = file_name.split('c')[0].split('t')[-1]
		prexy_name = file_name.split('c')[0].split('t')[-2]
		xy_name=prexy_name.split('xy')[-1]
		new_path = os.path.join(folder_path, xy_name)
		if not os.path.exists(new_path):
			os.makedirs(new_path)
		new_path_color = os.path.join(new_path, channel_name)
		if not os.path.exists(new_path_color):
			os.makedirs(new_path_color)
		old_image_path = os.path.join(folder_path, image)
		new_image_path = os.path.join(new_path_color,image)
		shutil.move(old_image_path, new_image_path)
	else:
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