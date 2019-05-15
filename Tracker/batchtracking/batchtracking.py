
# Batch automatic tracking

import os, sys
import os.path
import shutil
from itertools import groupby
from operator import itemgetter


from cell_ioC import *
from cell_drawing import prepare_mask_colors
from cell_trackingC import track_cells
from collections import defaultdict
# from skimage.morphology import binary_erosion
import logging


data_dir = "/home/paul/Desktop/20190410-2_T98G_Ni100_cytoMVIIC_MVIIC/EF"

# find all the subdirectory by first identifying all the directory in all levels, then add to a sub_directory list
#the intended prediction files should be organized by experiments into sets in a main folder which will be the direc_name

all_files = []
for root, dirs, files in os.walk(data_dir, topdown=False):
    for name in dirs:
        if str(name).find('_mask') != -1:
            all_files.append(os.path.join(root, name).replace("_mask","").replace("\\","/"))

# print(all_files)
# num_fileFind = len(all_files)


params = dict([('Raw image extension', ('tif', 'str')),
               ('Mask extension', ('png', 'str')),
               ('Mask folder suffix', ('_mask', 'str')),
               ('Show id\'s', (True, 'bool')),
               # ('Show contours', (True, 'bool')),
               ('Show tracks', (True, 'bool')),
               ('Pixel scale', (0.87e-6, 'float', True, 'm'))])
current_frame_index = -1
background_id = -1
raw_imgs = None
raw_masks = None
merged_masks = None
id_masks = None
id_masks_initial = None
cell_ids = []
cell_y = defaultdict(int)
cell_x = defaultdict(int)
max_cell_id = None
color_list = None
cell_color_idx = {}
trj = None
initial_trj = None
col_tuple = {}
col_weights = {}
show_ids = True
show_contours = True
show_tracks = True
cell_visibility = {}
cell_frame_presence = {}
cell_ids_raw_img = {}
cell_ids_mask = {}
# contour_data_per_frame = {}
# contour_plots_per_cell = {}
wide_track_cell_id = None
track_data_per_frame = defaultdict(lambda: np.zeros((0, 2)))
track_plots_per_cell = {}

def open_folder(path_in):
    global raw_imgs, raw_masks, merged_masks, id_masks, id_masks_initial, current_frame_index, \
        trj, cell_ids, background_id, color_list, wide_track_cell_id#, p_cell_selection
    # global path_in, raw_imgs, raw_masks, merged_masks, id_masks, id_masks_initial, current_frame_index, \
    #     trj, cell_ids, background_id, color_list, wide_track_cell_id
    
    # path_in = result
    print('Selected folder: "{}"'.format(path_in))
    # logging.info('Selected folder: "{}"'.format(path_in))
    new_imgs = read_img_sequence(path_in, params['Raw image extension'][0])
    new_masks = read_img_sequence(path_in + params['Mask folder suffix'][0],
                                    params['Mask extension'][0])
    if len(new_imgs) == len(new_masks) > 0:
        # Clear previous data
        trj = None
        col_tuple.clear()
        col_weights.clear()
        merged_masks = None
        id_masks = None
        id_masks_initial = None
        cell_ids = []
        cell_y.clear()
        cell_x.clear()
        # for cell_id in track_plots_per_cell:
        #     v_raw_img.removeItem(track_plots_per_cell[cell_id])
        #     # v_raw_img.removeItem(contour_plots_per_cell[cell_id])
        #     pi_raw_img.removeItem(cell_ids_raw_img[cell_id])
        #     pi_mask.removeItem(cell_ids_mask[cell_id])
        # contour_plots_per_cell.clear()
        # contour_data_per_frame.clear()
        wide_track_cell_id = None
        track_plots_per_cell.clear()
        track_data_per_frame.clear()
        color_list = None
        cell_color_idx.clear()
        cell_visibility.clear()
        cell_frame_presence.clear()
        cell_ids_raw_img.clear()
        cell_ids_mask.clear()
        # p_cell_selection.clearChildren()
        # v_raw_img.ui.histogram.gradient.restoreState(v_raw_img_original_state)
        # v_mask.ui.histogram.gradient.restoreState(v_mask_original_state)
        # Read new data
        raw_imgs = new_imgs
        raw_masks = new_masks
        current_frame_index = 0
        background_id = -1
        # v_raw_img.setImage(raw_imgs, axes={'x': 1, 'y': 0, 't': 2})
        # v_raw_img.setCurrentIndex(current_frame_index)
        # pi_raw_img.setWindowTitle(str(current_frame_index))
        # v_mask.setImage(raw_masks, axes={'x': 1, 'y': 0, 't': 2})
        # v_mask.setCurrentIndex(current_frame_index)
        # b_cell_tracking.setEnabled(True)
        # b_save_selected.setEnabled(False)
        # b_select_all.setEnabled(False)
        # b_select_none.setEnabled(False)
        # b_select_complete.setEnabled(False)
        # win.setWindowTitle('Cell Tracking ({})'.format(path_in))  # If you want to know which dataset you opened, pressing the "Open Folder" button will take you to the current dataset
    else:
        if len(new_imgs) == len(new_masks):
            print('Data reading failed. Failed to read data from path "{}"'.format(path_in))

        else:
            print('Data reading failed. The mask and raw image sequences have different lengths')

        return




def cell_tracking_clicked(path_in):
    global merged_masks, trj, initial_trj, col_tuple, col_weights, \
        cell_ids, max_cell_id, background_id, color_list, cell_color_idx, \
        cell_visibility, cell_frame_presence, id_masks, id_masks_initial
        #  p_cell_selection, id_masks, id_masks_initial
    merged_masks = raw_masks.copy()
# with pg.BusyCursor():
    # We set min_cell_id to 1 below because we need to reserve cell id 0 for the background, for plotting purposes
    trj, col_tuple, col_weights = track_cells(path_in, merged_masks, min_cell_id=1)
    initial_trj = trj.copy()
    id_masks, cell_ids, color_list, background_id = prepare_mask_colors(merged_masks, trj)
    cell_color_idx = dict([(x, col_idx) for col_idx, x in enumerate(cell_ids)])
    id_masks_initial = id_masks.copy()
    # Plot contours and tracks
    print('================= Calculating cell contours and tracks =================')
    for index, row in trj.sort_values('frame').iterrows():
        cell_id = row['particle']
        i_frame = row['frame']
        logging.info('Frame {}...'.format(i_frame))
        cell_y[cell_id, i_frame] = row['y']
        cell_x[cell_id, i_frame] = row['x']
        # # Calculate cell contours
        # cell_mask = (id_masks[:, :, i_frame] == cell_id).astype(np.uint8)
        # contour_data_per_frame[cell_id, i_frame] = \
        #     np.transpose(np.where(cell_mask - binary_erosion(cell_mask, selem=np.ones(3, 3)) > 0))

        # First we copy the latest available track of the cell
        # Here we take advantage that the tracks are sorted by frame,
        # so we know that the tracks in the previous frames were already included
        new_track = np.zeros((0, 2))
        for i_frame in range(int(row['frame']) - 1, -1, -1):
            if (row['particle'], i_frame) in track_data_per_frame:
                new_track = track_data_per_frame[row['particle'], i_frame].copy()
                break
        track_data_per_frame[row['particle'], row['frame']] = \
            np.append(new_track, np.array([(row['x'], row['y'])]), axis=0)
    # # Show cell id's
    cell_visibility = dict([(x, True) for x in cell_ids if x > 0])
    cell_frame_presence = trj.groupby('particle')['frame'].apply(set).to_dict()
    # # Create cell labels
    # for cell_id in cell_ids:
    #     if cell_id == background_id:
    #         continue
    #     color = color_list[cell_color_idx[cell_id]]
    #     # Cell labels in the raw image
    #     # curr_cell_id_img = pg.TextItem(str(cell_id), color=color_list[cell_color_idx[cell_id]], anchor=(0.5, 0.5))
    #     curr_cell_id_img = pg.TextItem(html='<div style="text-align: center"><b><span style="color: #{0:02x}{1:02x}{2:02x}; font-size: 8pt;">{3}</span></b></div>'.format(color[0], color[1], color[2], cell_id),
    #         anchor=(0, 0))
    #     pi_raw_img.addItem(curr_cell_id_img)
    #     curr_cell_id_img.setPos(cell_x[cell_id, current_frame_index], cell_y[cell_id, current_frame_index])
    #     cell_ids_raw_img[cell_id] = curr_cell_id_img
    #     # Cell labels in the mask
    #     # curr_cell_id_mask = pg.TextItem(str(cell_id), color=(255, 255, 255), anchor=(0.5, 0.5))
    #     curr_cell_id_mask = \
    #         pg.TextItem(html='<div style="text-align: center"><b><span style="color: #FFFFFF; font-size: 8pt;">{}</span></b></div>'.format(cell_id),
    #                     anchor=(0.5, 0.5))
    #     pi_mask.addItem(curr_cell_id_mask)
    #     curr_cell_id_mask.setPos(cell_x[cell_id, current_frame_index], cell_y[cell_id, current_frame_index])
    #     cell_ids_mask[cell_id] = curr_cell_id_mask
    #     # # Cell contours
    #     # contour_plots_per_cell[cell_id] = \
    #     #     pg.PlotDataItem(contour_data_per_frame[cell_id, current_frame_index],
    #     #                     pen=pg.mkPen(color_list[cell_color_idx[cell_id]], width=1))
    #     # v_raw_img.addItem(contour_plots_per_cell[cell_id])
    #     # Cell tracks
    #     track_plots_per_cell[cell_id] = \
    #         pg.PlotDataItem(track_data_per_frame[cell_id, current_frame_index],
    #                         pen=pg.mkPen(color_list[cell_color_idx[cell_id]], width=1))
    #     v_raw_img.addItem(track_plots_per_cell[cell_id])
    #     # Set cells visible or not in the current frame
    #     curr_cell_visibility = current_frame_index in cell_frame_presence[cell_id]
    #     curr_cell_id_img.setVisible(show_ids and curr_cell_visibility)
    #     curr_cell_id_mask.setVisible(show_ids and curr_cell_visibility)
    #     # contour_plots_per_cell[cell_id].setVisible(show_contours and curr_cell_visibility)
    #     track_plots_per_cell[cell_id].setVisible(show_tracks and curr_cell_visibility)
    # # Fill in right dock with checkboxes for each cell
    # p_cell_selection.addChildren(generate_cell_visibility_parametertree(cell_visibility))
    # pt_cell_selection.setParameters(p_cell_selection, showTop=False)
    # v_mask.setImage(id_masks, axes={'x': 1, 'y': 0, 't': 2})
    # max_cell_id = max(cell_ids)
    # v_mask.setColorMap(pg.ColorMap(pos=[x/max_cell_id for x in cell_ids], color=color_list))
    # v_mask.setCurrentIndex(current_frame_index)
    # b_save_selected.setEnabled(True)
    # b_select_all.setEnabled(True)
    # b_select_none.setEnabled(True)
    # b_select_complete.setEnabled(True)
    # pg.QtGui.QApplication.processEvents()  # Force redrawing of the canvas
    print('Cell tracking finished.')

# def select_all_clicked():
 
#     for child in p_cell_selection.children()[0]:
#         child.setValue(True)


# def select_none_clicked():

#     for child in p_cell_selection.children()[0]:
#         child.setValue(False)


# def select_complete_clicked():

#     n_frames = raw_imgs.shape[2]
#     for child in p_cell_selection.children()[0]:
#         cell_id = int(child.name())
#         visibility = (len(cell_frame_presence[cell_id]) == n_frames)
#         child.setValue(visibility)

def save_selected_clicked(path_in):
    if not any(cell_visibility.values()):
        print('Save failed. Please select at least one cell before saving')
        return
    
    save_results(path_in, trj, col_tuple, col_weights, id_masks, cell_ids, background_id,
                    color_list, cell_color_idx, cell_visibility, params['Pixel scale'][0], params['Pixel scale'][3],
                    params['Show id\'s'][0], True, params['Show tracks'][0],
                    params['Mask extension'][0])

## For test 
# print(all_files[0])
# open_folder(all_files[0])
# cell_tracking_clicked(all_files[0])
# save_selected_clicked(all_files[0])


## Run all the file
for x in range(0,len(all_files)):
    print(str(x+1)+'/' +str(len(all_files)))
    print(all_files[x])
    open_folder(all_files[x])
    cell_tracking_clicked(all_files[x])
    save_selected_clicked(all_files[x])
