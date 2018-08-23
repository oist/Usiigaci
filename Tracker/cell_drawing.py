# Author: ImagineA / Andrei Rares
# Date: 2018-08-18

import numpy as np
import colorsys
from random import shuffle
from PIL import Image, ImageDraw, ImageFont
from cell_tracking import get_trj_idx
from skimage.draw import line, line_aa
from skimage.morphology import binary_erosion
import logging


def prepare_mask_colors(merged_masks, trj):
    cell_ids = sorted(list(set(trj['particle'])))
    # background_id = 1 + cell_ids[-1]
    background_id = 0
    id_masks = create_id_masks(merged_masks, trj, background_id)
    cell_ids.insert(0, background_id)  # Insert background artificially
    color_list = generate_distinct_colors(n_colors=len(cell_ids), n_intensity_levels=3)
    color_list.insert(0, (20, 20, 20))  # Insert color for background (slightly brighter than black to see the limits)
    return id_masks, cell_ids, color_list, background_id


def create_id_masks(merged_masks, trj, background_id):
    id_masks = np.zeros_like(merged_masks) + background_id  # Fill with background
    for i_frame in range(merged_masks.shape[2]):
        new_frame = np.zeros(merged_masks.shape[:2], dtype=merged_masks.dtype)
        for id, intensity in trj.loc[trj['frame'] == i_frame, ('particle', 'mean_intensity')].values:
            new_frame[merged_masks[:, :, i_frame] == intensity] = id
        id_masks[:, :, i_frame] = new_frame
    return id_masks


def generate_distinct_colors(n_colors, n_intensity_levels = 2, max_channel_val=255):
    n_colors_per_intensity = int(np.ceil(n_colors / n_intensity_levels))
    RGB_tuples = []
    for intensity in np.arange(1, 0, -1 / n_intensity_levels):
        HSV_tuples = [(x/n_colors_per_intensity,
                       1,
                       1 - intensity/n_intensity_levels)
                      for x in range(n_colors_per_intensity)]
        RGB_tuples.extend([(int(np.floor(0.5 + x[0] * max_channel_val)),
                            int(np.floor(0.5 + x[1] * max_channel_val)),
                            int(np.floor(0.5 + x[2] * max_channel_val)))
                           for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))])
    shuffle(RGB_tuples)
    return RGB_tuples


def create_colorized_masks(id_masks, trj, cell_ids, background_id,
                           color_list, cell_color_idx,
                           cell_visibility, show_ids):
    logging.info('================= Colorizing masks =================')
    font = ImageFont.load_default()
    bg_color = color_list[cell_color_idx[background_id]]
    bg_frame = np.zeros((id_masks.shape[0], id_masks.shape[1], 3), dtype=id_masks.dtype)
    for color_channel in range(3):
        bg_frame[:, :, color_channel] += bg_color[color_channel]
    colorized_masks = []
    for i_frame in range(id_masks.shape[2]):
        logging.info('Frame {}...'.format(i_frame))
        mask = id_masks[:, :, i_frame]
        col_frame = bg_frame.copy()
        if show_ids:
            id_frame = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=mask.dtype)
        for cell_id in cell_ids:
            if (cell_id == background_id) or (not cell_visibility[cell_id]):
                continue
            cell_idx = get_trj_idx(trj, i_frame, 'particle', [cell_id])
            if len(cell_idx) == 0:
                continue
            cell_y = int(trj.loc[cell_idx[0], 'y'])
            cell_x = int(trj.loc[cell_idx[0], 'x'])
            cell_color = color_list[cell_color_idx[cell_id]]
            cell_coords = (mask == cell_id)
            for color_channel in range(3):
                col_frame[:, :, color_channel][cell_coords] = cell_color[color_channel]
            if show_ids:
                # Stamp the cell_id on cell
                draw_img = Image.fromarray(id_frame, 'RGB')
                draw_text = ImageDraw.Draw(draw_img)
                id_txt = str(cell_id)
                id_width, id_height = font.getsize(id_txt)
                draw_text.text((min(cell_x, mask.shape[1] - id_width),
                                max(0, cell_y - id_height)), id_txt,
                               (255, 255, 255),
                               font=font)
                id_frame = np.asarray(draw_img).copy()
        if show_ids:
            # Put the id's on top of the cells
            col_frame[id_frame > 0] = 255
        colorized_masks.append(col_frame)
    return colorized_masks


def create_colorized_tracks(id_masks, trj, cell_ids, background_id,
                            color_list, cell_color_idx,
                            cell_visibility, show_ids, show_contours, show_tracks,
                            use_thick_line):
    logging.info('================= Colorizing tracks =================')
    font = ImageFont.load_default()
    track_accumulator_y = {}
    track_accumulator_x = {}
    structure_element = np.ones((5, 5)) if use_thick_line else np.ones((3, 3))
    colorized_tracks = []
    for i_frame in range(id_masks.shape[2]):
        logging.info('Frame {}...'.format(i_frame))
        mask = id_masks[:, :, i_frame]
        col_frame = np.stack((mask.copy(), mask.copy(), mask.copy()), axis=2)
        if show_ids:
            id_frame = np.zeros_like(col_frame)
        for cell_id in cell_ids:
            if (cell_id == background_id) or (not cell_visibility[cell_id]):
                continue
            cell_idx = get_trj_idx(trj, i_frame, 'particle', [cell_id])
            if len(cell_idx) == 0:
                continue
            cell_y = int(trj.loc[cell_idx[0], 'y'])
            cell_x = int(trj.loc[cell_idx[0], 'x'])
            cell_color = color_list[cell_color_idx[cell_id]]
            if show_tracks:
                # Compute cell's last displacement (aka "leg")
                if cell_id not in track_accumulator_y:
                    track_accumulator_y[cell_id] = np.array([cell_y], dtype=np.int64)
                    track_accumulator_x[cell_id] = np.array([cell_x], dtype=np.int64)
                else:
                    if use_thick_line:
                        last_leg_y, last_leg_x, _ = line_aa(track_accumulator_y[cell_id][-1],
                                                            track_accumulator_x[cell_id][-1],
                                                            cell_y,
                                                            cell_x)
                    else:
                        last_leg_y, last_leg_x, _ = line(track_accumulator_y[cell_id][-1],
                                                         track_accumulator_x[cell_id][-1],
                                                         cell_y,
                                                         cell_x)
                    track_accumulator_y[cell_id] = np.concatenate((track_accumulator_y[cell_id], last_leg_y),
                                                                  axis=0)
                    track_accumulator_x[cell_id] = np.concatenate((track_accumulator_x[cell_id], last_leg_x),
                                                                  axis=0)
                # Draw cell track
                for i_color in range(3):
                    col_frame[:, :, i_color][track_accumulator_y[cell_id],
                                             track_accumulator_x[cell_id]] = cell_color[i_color]
            if show_contours:
                # Compute cell border
                cell_mask = (mask == cell_id).astype(np.uint8)
                inner_cell_border = np.where(cell_mask - binary_erosion(cell_mask, selem=structure_element) > 0)
                # Draw cell border
                for i_color in range(3):
                    col_frame[:, :, i_color][inner_cell_border] = cell_color[i_color]
            if show_ids:
                # Stamp the cell_id on cell
                draw_img = Image.fromarray(id_frame, 'RGB')
                draw_text = ImageDraw.Draw(draw_img)
                id_txt = str(cell_id)
                id_width, id_height = font.getsize(id_txt)
                draw_text.text((min(cell_x, mask.shape[1] - id_width),
                                max(0, cell_y - id_height)), id_txt,
                               cell_color,
                               font=font)
                id_frame = np.asarray(draw_img).copy()
        if show_ids:
            # Put the id's on top of the cells
            col_frame[id_frame > 0] = id_frame[id_frame > 0]
        colorized_tracks.append(col_frame)
    return colorized_tracks


def create_track_overview(id_masks, trj, cell_ids, background_id,
                          color_list, cell_color_idx,
                          cell_visibility, show_ids,
                          use_thick_line):
    logging.info('================= Creating track overview =================')
    track_accumulator_y = {}
    track_accumulator_x = {}
    for i_frame in range(id_masks.shape[2]):
        logging.info('Frame {}...'.format(i_frame))
        for cell_id in cell_ids:
            if (cell_id == background_id) or (not cell_visibility[cell_id]):
                continue
            cell_idx = get_trj_idx(trj, i_frame, 'particle', [cell_id])
            if len(cell_idx) == 0:
                continue
            cell_y = int(trj.loc[cell_idx[0], 'y'])
            cell_x = int(trj.loc[cell_idx[0], 'x'])
            # Compute cell's last displacement (aka "leg")
            if cell_id not in track_accumulator_y:
                track_accumulator_y[cell_id] = np.array([cell_y], dtype=np.int64)
                track_accumulator_x[cell_id] = np.array([cell_x], dtype=np.int64)
            else:
                if use_thick_line:
                    last_leg_y, last_leg_x, _ = line_aa(track_accumulator_y[cell_id][-1],
                                                        track_accumulator_x[cell_id][-1],
                                                        cell_y,
                                                        cell_x)
                else:
                    last_leg_y, last_leg_x, _ = line(track_accumulator_y[cell_id][-1],
                                                     track_accumulator_x[cell_id][-1],
                                                     cell_y,
                                                     cell_x)
                track_accumulator_y[cell_id] = np.concatenate((track_accumulator_y[cell_id], last_leg_y),
                                                              axis=0)
                track_accumulator_x[cell_id] = np.concatenate((track_accumulator_x[cell_id], last_leg_x),
                                                              axis=0)
    track_overview = np.stack((id_masks[:, :, -1].copy(),
                               id_masks[:, :, -1].copy(),
                               id_masks[:, :, -1].copy()), axis=2)
    # Add calculated tracks and cell id's to the final frame
    if show_ids:
        id_frame = np.zeros_like(track_overview)
    font = ImageFont.load_default()
    for cell_id in cell_ids:
        if (cell_id == background_id) or (not cell_visibility[cell_id]):
            continue
        last_cell_frame = trj.groupby('particle')['frame'].apply(max).to_dict()
        cell_idx = get_trj_idx(trj, last_cell_frame[cell_id], 'particle', [cell_id])
        if len(cell_idx) == 0:
            continue
        cell_y = int(trj.loc[cell_idx[0], 'y'])
        cell_x = int(trj.loc[cell_idx[0], 'x'])
        cell_color = color_list[cell_color_idx[cell_id]]
        for i_color in range(3):
            # Draw cell track
            track_overview[:, :, i_color][track_accumulator_y[cell_id],
                                          track_accumulator_x[cell_id]] = cell_color[i_color]
        if show_ids:
            # Stamp the cell_id on cell
            draw_img = Image.fromarray(id_frame, 'RGB')
            draw_text = ImageDraw.Draw(draw_img)
            id_txt = str(cell_id)
            id_width, id_height = font.getsize(id_txt)
            draw_text.text((min(cell_x, id_masks.shape[1] - id_width),
                            max(0, cell_y - id_height)), id_txt,
                           cell_color,
                           font=font)
            id_frame = np.asarray(draw_img).copy()
    if show_ids:
        # Put the id's on top of the cells
        track_overview[id_frame > 0] = id_frame[id_frame > 0]
    return track_overview


