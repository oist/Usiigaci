# Author: ImagineA / Andrei Rares
# Date: 2018-08-18

import pims
import numpy as np
from os.path import join, split, normpath, exists
from os import makedirs
from datetime import datetime
from cell_drawing import create_colorized_masks, create_colorized_tracks, create_track_overview
from imageio import imwrite
import cv2
import logging
import png


def read_img_sequence(path, file_extension):
    pims_sequence = pims.ImageSequence(join(path, '*.{}'.format(file_extension)), process_func=None)
    # return np.stack([frame.copy() for frame in pims_sequence], axis=2)
    return np.stack([frame.astype(np.uint16) for frame in pims_sequence], axis=2)


def save_results(path_in, trj, col_tuple, col_weights, id_masks, cell_ids, background_id,
                 color_list, cell_color_idx, cell_visibility, pixel_scale, pixel_unit,
                 show_ids, show_contours, show_tracks,
                 mask_extension):
    # Create output folder based on input path and experiment date/time
    save_time = datetime.now()  # Experiment time
    root_path, folder_in = split(normpath(path_in))
    path_out = join(root_path,
                    '{}_Exp_{}-{:02d}-{:02d}T{:02d}{:02d}{:02d}'.format(
                        folder_in,
                        save_time.year, save_time.month, save_time.day,
                        save_time.hour, save_time.minute, save_time.second))
    if not exists(path_out):
        makedirs(path_out)

    pixel_scale *= 10**6  # Force micrometer scale
    # Save CSV results
    save_results_to_csv(path_out, trj, col_tuple, cell_visibility, pixel_scale)

    # Save experiment parameters
    save_experiment_parameters(path_out, pixel_scale, pixel_unit, col_weights)

    # Save id masks (may be useful for later postprocessing
    logging.info('================= Saving id masks frame by frame =================')
    save_sequence_frame_by_frame([id_masks[:, :, i_frame] for i_frame in range(id_masks.shape[2])],
                                 path_out, 'Id_masks_per_frame', mask_extension, 'id_masks')

    # Save colorized masks
    colorized_masks = create_colorized_masks(id_masks, trj, cell_ids, background_id,
                                             color_list, cell_color_idx,
                                             cell_visibility, show_ids)
    logging.info('================= Saving colorized masks frame by frame =================')
    save_sequence_frame_by_frame(colorized_masks, path_out, 'Masks_per_frame', mask_extension, 'masks')
    logging.info('================= Saving colorized masks as animation =================')
    # mimwrite(join(path_out, 'masks_animation.mp4'), colorized_masks, macro_block_size=None)
    cm_movie_writer = cv2.VideoWriter(
        join(path_out, 'masks_animation.mp4'),
        cv2.VideoWriter_fourcc(*'MP4V'),
        5.0,
        (colorized_masks[0].shape[1], colorized_masks[0].shape[0]))  # (width, height)
    for mask in colorized_masks:
        cm_movie_writer.write(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    cm_movie_writer.release()

    # Save colorized tracks
    colorized_tracks = create_colorized_tracks(id_masks, trj, cell_ids, background_id,
                                               color_list, cell_color_idx,
                                               cell_visibility, show_ids, show_contours, show_tracks,
                                               True)
    logging.info('================= Saving colorized tracks frame by frame =================')
    save_sequence_frame_by_frame(colorized_tracks, path_out, 'Tracks_per_frame', mask_extension, 'tracks')
    logging.info('================= Saving colorized tracks as animation =================')
    # mimwrite(join(path_out, 'tracks_animation.mp4'), colorized_tracks, macro_block_size=None)
    tr_movie_writer = cv2.VideoWriter(
        join(path_out, 'tracks_animation.mp4'),
        cv2.VideoWriter_fourcc(*'MP4V'),
        5.0,
        (colorized_tracks[0].shape[1], colorized_tracks[0].shape[0]))  # (width, height)
    for mask in colorized_tracks:
        tr_movie_writer.write(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    tr_movie_writer.release()

    # Save complete tracks
    logging.info('================= Saving track overview =================')
    all_tracks = create_track_overview(id_masks, trj, cell_ids, background_id,
                                       color_list, cell_color_idx,
                                       cell_visibility, show_ids,
                                       True)
    cv2.imwrite(join(path_out, 'all_tracks.{}'.format(mask_extension)), cv2.cvtColor(all_tracks, cv2.COLOR_RGB2BGR))
    logging.info('Saving finished.')


def save_results_to_csv(path_out, trj, col_tuple, cell_visibility, pixel_scale):
    cols_to_save = ['particle'] + col_tuple['original'] + col_tuple['weighted'] + col_tuple['extra']
    scaled_cols = {'y': pixel_scale,
                   'x': pixel_scale,
                   'equivalent_diameter': pixel_scale,
                   'perimeter': pixel_scale,
                   'area': pixel_scale * pixel_scale}  # The area must be proportional to the square of the scale
    order_list = ['particle', 'frame']
    # Make a scaled copy before saving to file
    scaled_trj = trj.copy()
    for col_name, col_scale in scaled_cols.items():
        scaled_trj[col_name] = col_scale * scaled_trj[col_name]
    scaled_trj[scaled_trj['particle'].isin([cell_id
                                            for cell_id, show_cell in cell_visibility.items()
                                            if show_cell])].sort_values(order_list).to_csv(
        join(path_out, 'tracks.csv'),
        columns=cols_to_save,
        float_format='%f',  # Use '%.03f' for 3 digits after the comma
        index=False)


def save_experiment_parameters(path_out, pixel_scale, pixel_unit, col_weights):
    out_lines = ['Parameter,Value\n',
                 '{},{}\n'.format('pixel_scale', pixel_scale),
                 '{},{}\n'.format('pixel_unit', 'microns')]  # Normally we should have had pixel_unit instead of 'microns', but we multiplied pixel_scale by 1 million in the caller function
    for param_name, param_value in col_weights.items():
        out_lines.append('{},{}\n'.format(param_name, param_value))
    with open(join(path_out, 'experiment_parameters.csv'), 'w') as f_out:
        f_out.writelines(out_lines)


def save_sequence_frame_by_frame(sequence, path_out, sequence_folder, file_extension, file_prefix):
    path_out_sequence = join(path_out, sequence_folder)
    if not exists(path_out_sequence):
        makedirs(path_out_sequence)
    n_frames = len(sequence)
    max_n_digits = 1 + int(np.floor(np.log10(n_frames - 1)))
    if sequence[0].dtype == np.uint8:
        for i_frame, frame in enumerate(sequence):
            logging.info('Frame {}...'.format(i_frame))
            frame_name = '{}_{}.{}'.format(file_prefix, str(i_frame).zfill(max_n_digits), file_extension)
            file_name = join(path_out_sequence, frame_name)
            if frame.ndim == 3:
                cv2.imwrite(file_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(file_name, frame)
    elif sequence[0].dtype == np.uint16:
        if sequence[0].ndim == 3:
            writer = png.Writer(width=sequence[0].shape[1], height=sequence[0].shape[0], bitdepth=16)
        for i_frame, frame in enumerate(sequence):
            logging.info('Frame {}...'.format(i_frame))
            frame_name = '{}_{}.{}'.format(file_prefix, str(i_frame).zfill(max_n_digits), file_extension)
            file_name = join(path_out_sequence, frame_name)
            if frame.ndim == 3:
                with open(file_name, 'wb') as f_out:
                    writer.write(f_out, frame.reshape(-1, frame.shape[1]*frame.shape[2]).tolist())
            else:
                imwrite(file_name, frame)

