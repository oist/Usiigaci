# Author: ImagineA / Andrei Rares
# Date: 2018-08-18

import pyqtgraph as pg
from pyqtgraph.dockarea import *
from PyQt5.QtWidgets import QFileDialog
from cell_io import *
from cell_img_proc import prepare_mask_colors
from cell_tracking import track_cells
from collections import defaultdict
# from skimage.morphology import binary_erosion
import logging


# Custom logging format
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')


# Pyqtgraph's ImageView does not emit “time changed” signal
# https://stackoverflow.com/questions/32586149/
class PatchedImageView(pg.ImageView):
    def timeLineChanged(self):
        (ind, time) = self.timeIndex(self.timeLine)
        self.sigTimeChanged.emit(ind, time)
        if self.ignoreTimeLine:
            return
        self.play(0)
        if ind != self.currentIndex:
            self.currentIndex = ind
            self.updateImage()


def open_folder_clicked(param):
    global path_in, raw_imgs, raw_masks, merged_masks, id_masks, id_masks_initial, current_frame_index, \
        trj, cell_ids, background_id, color_list, wide_track_cell_id
    result = QFileDialog.getExistingDirectory(None, 'Select Folder', path_in)
    if len(result) == 0:
        logging.info('Canceled folder selection')
    else:
        path_in = result
        logging.info('Selected folder: "{}"'.format(path_in))
        new_imgs = read_img_sequence(path_in, params['Raw image extension'][0])
        new_masks = read_img_sequence(path_in + params['Mask folder suffix'][0],
                                      params['Mask extension'][0])
        if len(new_imgs) == len(new_masks) > 0:
            # Clear previous data
            trj = None
            col_tuple.clear()
            merged_masks = None
            id_masks = None
            id_masks_initial = None
            cell_ids = []
            cell_y.clear()
            cell_x.clear()
            for cell_id in track_plots_per_cell:
                v_raw_img.removeItem(track_plots_per_cell[cell_id])
                # v_raw_img.removeItem(contour_plots_per_cell[cell_id])
                pi_raw_img.removeItem(cell_ids_raw_img[cell_id])
                pi_mask.removeItem(cell_ids_mask[cell_id])
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
            p_cell_selection.clearChildren()
            v_raw_img.ui.histogram.gradient.restoreState(v_raw_img_original_state)
            v_mask.ui.histogram.gradient.restoreState(v_mask_original_state)
            # Read new data
            raw_imgs = new_imgs
            raw_masks = new_masks
            current_frame_index = 0
            background_id = -1
            v_raw_img.setImage(raw_imgs, axes={'x': 1, 'y': 0, 't': 2})
            v_raw_img.setCurrentIndex(current_frame_index)
            pi_raw_img.setWindowTitle(str(current_frame_index))
            v_mask.setImage(raw_masks, axes={'x': 1, 'y': 0, 't': 2})
            v_mask.setCurrentIndex(current_frame_index)
            b_cell_tracking.setEnabled(True)
            b_save_selected.setEnabled(False)
            b_select_all.setEnabled(False)
            b_select_none.setEnabled(False)
            b_select_complete.setEnabled(False)
        else:
            logging.error('Failed to read data from path "{}"'.format(path_in))
            return


def cell_tracking_clicked(param):
    global merged_masks, trj, initial_trj, col_tuple, \
        cell_ids, max_cell_id, background_id, color_list, cell_color_idx, \
        cell_visibility, cell_frame_presence, p_cell_selection, id_masks, id_masks_initial
    merged_masks = raw_masks.copy()
    with pg.BusyCursor():
        # We set min_cell_id to 1 below because we need to reserve cell id 0 for the background, for plotting purposes
        trj, col_tuple = track_cells(path_in, merged_masks, min_cell_id=1)
        initial_trj = trj.copy()
        id_masks, cell_ids, color_list, background_id = prepare_mask_colors(merged_masks, trj)
        cell_color_idx = dict([(x, col_idx) for col_idx, x in enumerate(cell_ids)])
        id_masks_initial = id_masks.copy()
        # Plot contours and tracks
        logging.info('================= Calculating cell contours and tracks =================')
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
        # Show cell id's
        cell_visibility = dict([(x, True) for x in cell_ids if x > 0])
        cell_frame_presence = trj.groupby('particle')['frame'].apply(set).to_dict()
        # Create cell labels
        for cell_id in cell_ids:
            if cell_id == background_id:
                continue
            color = color_list[cell_color_idx[cell_id]]
            # Cell labels in the raw image
            # curr_cell_id_img = pg.TextItem(str(cell_id), color=color_list[cell_color_idx[cell_id]], anchor=(0.5, 0.5))
            curr_cell_id_img = pg.TextItem(html='<div style="text-align: center"><b><span style="color: #{0:02x}{1:02x}{2:02x}; font-size: 8pt;">{3}</span></b></div>'.format(color[0], color[1], color[2], cell_id),
                anchor=(0, 0))
            pi_raw_img.addItem(curr_cell_id_img)
            curr_cell_id_img.setPos(cell_x[cell_id, current_frame_index], cell_y[cell_id, current_frame_index])
            cell_ids_raw_img[cell_id] = curr_cell_id_img
            # Cell labels in the mask
            # curr_cell_id_mask = pg.TextItem(str(cell_id), color=(255, 255, 255), anchor=(0.5, 0.5))
            curr_cell_id_mask = \
                pg.TextItem(html='<div style="text-align: center"><b><span style="color: #FFFFFF; font-size: 8pt;">{}</span></b></div>'.format(cell_id),
                            anchor=(0.5, 0.5))
            pi_mask.addItem(curr_cell_id_mask)
            curr_cell_id_mask.setPos(cell_x[cell_id, current_frame_index], cell_y[cell_id, current_frame_index])
            cell_ids_mask[cell_id] = curr_cell_id_mask
            # # Cell contours
            # contour_plots_per_cell[cell_id] = \
            #     pg.PlotDataItem(contour_data_per_frame[cell_id, current_frame_index],
            #                     pen=pg.mkPen(color_list[cell_color_idx[cell_id]], width=1))
            # v_raw_img.addItem(contour_plots_per_cell[cell_id])
            # Cell tracks
            track_plots_per_cell[cell_id] = \
                pg.PlotDataItem(track_data_per_frame[cell_id, current_frame_index],
                                pen=pg.mkPen(color_list[cell_color_idx[cell_id]], width=1))
            v_raw_img.addItem(track_plots_per_cell[cell_id])
            # Set cells visible or not in the current frame
            curr_cell_visibility = current_frame_index in cell_frame_presence[cell_id]
            curr_cell_id_img.setVisible(show_ids and curr_cell_visibility)
            curr_cell_id_mask.setVisible(show_ids and curr_cell_visibility)
            # contour_plots_per_cell[cell_id].setVisible(show_contours and curr_cell_visibility)
            track_plots_per_cell[cell_id].setVisible(show_tracks and curr_cell_visibility)
        # Fill in right dock with checkboxes for each cell
        p_cell_selection.addChildren(generate_cell_visibility_parametertree(cell_visibility))
        pt_cell_selection.setParameters(p_cell_selection, showTop=False)
        v_mask.setImage(id_masks, axes={'x': 1, 'y': 0, 't': 2})
        max_cell_id = max(cell_ids)
        v_mask.setColorMap(pg.ColorMap(pos=[x/max_cell_id for x in cell_ids], color=color_list))
        v_mask.setCurrentIndex(current_frame_index)
        b_save_selected.setEnabled(True)
        b_select_all.setEnabled(True)
        b_select_none.setEnabled(True)
        b_select_complete.setEnabled(True)
        pg.QtGui.QApplication.processEvents()  # Force redrawing of the canvas
        logging.info('Cell tracking finished.')


def generate_cell_visibility_parametertree(cell_visibility):
    return [{'name': 'Cell list', 'type': 'group',
             'children': [{'name': str(x), 'type': 'bool', 'value': y}
                          for x, y in cell_visibility.items()]}]


def save_selected_clicked(param):
    if not any(cell_visibility.values()):
        pg.Qt.QtGui.QMessageBox.warning(b_save_selected,
                                        'Save failed',
                                        'Please select at least one cell before saving')
        return
    with pg.BusyCursor():
        save_results(path_in, trj, col_tuple, id_masks, cell_ids, background_id,
                     color_list, cell_color_idx, cell_visibility,
                     params['Show id\'s'][0], True, params['Show tracks'][0],
                     params['Mask extension'][0])


def select_all_clicked(param):
    with pg.BusyCursor():
        for child in p_cell_selection.children()[0]:
            child.setValue(True)


def select_none_clicked(param):
    with pg.BusyCursor():
        for child in p_cell_selection.children()[0]:
            child.setValue(False)


def select_complete_clicked(param):
    with pg.BusyCursor():
        n_frames = raw_imgs.shape[2]
        for child in p_cell_selection.children()[0]:
            cell_id = int(child.name())
            visibility = (len(cell_frame_presence[cell_id]) == n_frames)
            child.setValue(visibility)



# Frame number will change at integer values, but signals will be also issued at non-integer positions
def time_changed_raw_img(param, value):
    global current_frame_index
    if v_raw_img.currentIndex != current_frame_index:
        current_frame_index = v_raw_img.currentIndex
        pi_raw_img.setWindowTitle(str(current_frame_index))
        v_mask.setCurrentIndex(current_frame_index)
        for cell_id in cell_ids:
            if cell_id == background_id:
                continue
            cell_ids_raw_img[cell_id].setPos(cell_x[cell_id, current_frame_index],
                                             cell_y[cell_id, current_frame_index])
            cell_ids_mask[cell_id].setPos(cell_x[cell_id, current_frame_index],
                                          cell_y[cell_id, current_frame_index])
            track_plots_per_cell[cell_id].setData(track_data_per_frame[cell_id, current_frame_index])
            # Set cells visible or not
            curr_visibility = cell_visibility[cell_id] \
                              and (current_frame_index in cell_frame_presence[cell_id])
            cell_ids_raw_img[cell_id].setVisible(show_ids and curr_visibility)
            cell_ids_mask[cell_id].setVisible(show_ids and curr_visibility)
            # contour_plots_per_cell[cell_id].setVisible(show_contours and curr_visibility)
            track_plots_per_cell[cell_id].setVisible(show_tracks and curr_visibility)



# Frame number will change at integer values, but signals will be also issued at non-integer positions
def time_changed_mask(param, value):
    global current_frame_index
    if v_mask.currentIndex != current_frame_index:
        current_frame_index = v_mask.currentIndex
        pi_mask.setWindowTitle(str(current_frame_index))
        v_raw_img.setCurrentIndex(current_frame_index)
        for cell_id in cell_ids:
            if cell_id == background_id:
                continue
            cell_ids_raw_img[cell_id].setPos(cell_x[cell_id, current_frame_index],
                                             cell_y[cell_id, current_frame_index])
            cell_ids_mask[cell_id].setPos(cell_x[cell_id, current_frame_index],
                                          cell_y[cell_id, current_frame_index])
            track_plots_per_cell[cell_id].setData(track_data_per_frame[cell_id, current_frame_index])
            # Set cells visible or not
            curr_visibility = cell_visibility[cell_id] \
                              and (current_frame_index in cell_frame_presence[cell_id])
            cell_ids_raw_img[cell_id].setVisible(show_ids and curr_visibility)
            cell_ids_mask[cell_id].setVisible(show_ids and curr_visibility)
            # contour_plots_per_cell[cell_id].setVisible(show_contours and curr_visibility)
            track_plots_per_cell[cell_id].setVisible(show_tracks and curr_visibility)


def param_changed(param, value):
    global show_ids, show_contours, show_tracks
    param_type = params[param.name()][1]
    params[param.name()] = (value, param_type)
    if param.name() == 'Show id\'s':
        show_ids = value
        for cell_id in cell_ids_raw_img:
            cell_ids_raw_img[cell_id].setVisible(show_ids and cell_visibility[cell_id])
            cell_ids_mask[cell_id].setVisible(show_ids and cell_visibility[cell_id])
    # elif param.name() == 'Show contours':
    #     show_contours = value
    #     # for cell_id in contour_plots_per_cell:
    #     #     contour_plots_per_cell[cell_id].setVisible(show_contours and cell_visibility[cell_id])
    elif param.name() == 'Show tracks':
        show_tracks = value
        for cell_id in track_plots_per_cell:
            track_plots_per_cell[cell_id].setVisible(show_tracks and cell_visibility[cell_id])



def cell_selection_changed(param, changes):
    for cell_param, change, value in changes:
        path = p_cell_selection.childPath(cell_param)
        if path is not None and len(path) > 0:
            cell_id = int(path[-1])
            update_cell_visibility(cell_id, value)


def update_cell_visibility(cell_id, visible):
    cell_visibility[cell_id] = visible
    if visible:
        # Recover original position and value of the cell in the mask
        id_masks[id_masks_initial == cell_id] = cell_id
    else:
        # Hide cell by giving it the id of the background
        id_masks[id_masks == cell_id] = 0
    cell_ids_raw_img[cell_id].setVisible(show_ids and visible)
    cell_ids_mask[cell_id].setVisible(show_ids and visible)
    # contour_plots_per_cell[cell_id].setVisible(show_contours and visible)
    track_plots_per_cell[cell_id].setVisible(show_tracks and visible)
    v_mask.updateImage()


def cell_focus_changed(param, column):
    global wide_track_cell_id
    cell_id = int(param.param.name())
    if cell_id != wide_track_cell_id:
        if wide_track_cell_id is not None:
            # Make previous track thin again
            track_plots_per_cell[wide_track_cell_id].setPen(
                pg.mkPen(color_list[cell_color_idx[wide_track_cell_id]], width=1))
        track_plots_per_cell[cell_id].setPen(
            pg.mkPen(color_list[cell_color_idx[cell_id]], width=5))
        wide_track_cell_id = cell_id


# Experiment related vars
path_in = r'../../Data/'
params = dict([('Raw image extension', ('tif', 'str')),
               ('Mask extension', ('png', 'str')),
               ('Mask folder suffix', ('_mask_avg', 'str')),
               ('Show id\'s', (True, 'bool')),
               # ('Show contours', (True, 'bool')),
               ('Show tracks', (True, 'bool'))])
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
# GUI related vars
app = pg.Qt.QtGui.QApplication([])
win = pg.Qt.QtGui.QMainWindow()
dock_area = DockArea()
win.setCentralWidget(dock_area)
win.resize(1000, 600)
win.setWindowTitle('Cell Tracking')

## Create docks
d_io = Dock("I/O", size=(1, 1))  # I/O dock
d_img = Dock("Img", size=(800, 400))  # Image dock
d_tracks = Dock("Tracks", size=(150, 400))  # Tracks dock

dock_area.addDock(d_io, 'top')
dock_area.addDock(d_img, 'bottom')
dock_area.addDock(d_tracks, 'right')

## Add widgets into each dock

# Initialize I/O button laout
l_io = pg.LayoutWidget()
# I/O file extension parameters
pt_io_params = pg.parametertree.ParameterTree(showHeader=False)
for param_name, (param_val, param_type) in params.items():
    p_new = pg.parametertree.Parameter.create(name=param_name, type=param_type, value=param_val)
    p_new.sigValueChanged.connect(param_changed)
    pt_io_params.addParameters(p_new)
l_io.addWidget(pt_io_params, row=0, col=0)
l_selection = pg.LayoutWidget()
l_io.addWidget(l_selection, row=0, col=1, rowspan=2, colspan=3)
# Open folder button
b_open_folder = pg.Qt.QtGui.QPushButton("Open\nFolder")
b_open_folder.clicked.connect(open_folder_clicked)
l_selection.addWidget(b_open_folder, row=0, col=0)
# Cell tracking button
b_cell_tracking = pg.QtGui.QPushButton("Cell\nTracking")
b_cell_tracking.clicked.connect(cell_tracking_clicked)
b_cell_tracking.setEnabled(False)
l_selection.addWidget(b_cell_tracking, row=0, col=1)
# Save selected tracks button
b_save_selected = pg.QtGui.QPushButton("Save\nSelection")
b_save_selected.clicked.connect(save_selected_clicked)
b_save_selected.setEnabled(False)
l_selection.addWidget(b_save_selected, row=0, col=2)
# Select all tracks button
b_select_all = pg.QtGui.QPushButton("Select\nAll Tracks")
b_select_all.clicked.connect(select_all_clicked)
b_select_all.setEnabled(False)
l_selection.addWidget(b_select_all, row=2, col=0)
# Select no tracks button
b_select_none = pg.QtGui.QPushButton("Select\nNone")
b_select_none.clicked.connect(select_none_clicked)
b_select_none.setEnabled(False)
l_selection.addWidget(b_select_none, row=2, col=1)
# Select complete tracks button
b_select_complete = pg.QtGui.QPushButton("Select\nComplete Tracks")
b_select_complete.clicked.connect(select_complete_clicked)
b_select_complete.setEnabled(False)
l_selection.addWidget(b_select_complete, row=2, col=2)
# Add I/O buttons to the docking area
d_io.addWidget(l_io, row=0, col=0)

# Initialize raw image viewer
pi_raw_img = pg.PlotItem()
# pi_raw_img.getViewBox().menu.ctrl[0].mouseCheck.setChecked(False)  # Disable horizontal pan
# pi_raw_img.getViewBox().menu.ctrl[1].mouseCheck.setChecked(False)  # Disable vertical pan
v_raw_img = PatchedImageView(view=pi_raw_img)
v_raw_img.sigTimeChanged.connect(time_changed_raw_img)
v_raw_img.setImage(np.random.normal(size=(100, 100)))
v_raw_img.view.invertY(True)
v_raw_img.ui.histogram.hide()
v_raw_img.ui.menuBtn.hide()
v_raw_img.ui.roiBtn.hide()
v_raw_img_original_state = v_raw_img.ui.histogram.gradient.saveState()
# Add image to the docking area
d_img.addWidget(v_raw_img, row=0, col=0, rowspan=2, colspan=2)

# Initialize mask viewer
pi_mask = pg.PlotItem()
# pi_mask.getViewBox().menu.ctrl[0].mouseCheck.setChecked(False)  # Disable horizontal pan
# pi_mask.getViewBox().menu.ctrl[1].mouseCheck.setChecked(False)  # Disable vertical pan
v_mask = PatchedImageView(view=pi_mask)
v_mask.sigTimeChanged.connect(time_changed_mask)
v_mask.setImage(np.random.normal(size=(100, 100)))
v_mask.view.invertY(True)
v_mask.ui.histogram.hide()
v_mask.ui.menuBtn.hide()
v_mask.ui.roiBtn.hide()
v_mask_original_state = v_mask.ui.histogram.gradient.saveState()
# Add image to the docking area
d_img.addWidget(v_mask, row=0, col=2, rowspan=2, colspan=2)

# Initialize cell selection GUI
p_cell_selection = pg.parametertree.Parameter.create(name='cell selection', type='group',
                                                     children=generate_cell_visibility_parametertree(
                                                         cell_visibility))
p_cell_selection.sigTreeStateChanged.connect(cell_selection_changed)
pt_cell_selection = pg.parametertree.ParameterTree()
pt_cell_selection.setParameters(p_cell_selection, showTop=False)
pt_cell_selection.itemClicked.connect(cell_focus_changed)
# Add track list to the docking area
d_tracks.addWidget(pt_cell_selection, row=0, col=0)

# Show main window
win.show()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()
