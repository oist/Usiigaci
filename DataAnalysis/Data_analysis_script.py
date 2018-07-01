'''
Single cell tracking data processing script
Hsieh-Fu Tsai (hsiehfutsai@gmail.com), Tyler Sloan(info@quorumetrix.com), Amy Shen(amy.shen@oist.jp)

purpose:
this notebook aims to be a general tool for analysis of single cell migration data with use of opensource tools.

Input data:
the script can process cell tracking data from ImageJ, Lineage mapper, or Imaris.
If you use this code, please cite the following paper:
Hsieh-Fu Tsai, Tyler Sloan, Joanna Gajda, Amy Shen, Usiigaci: Label-free instance-aware cell tracking in phase contrast microscopy using Mask R-CNN.

License:
This script is released under MIT license
Copyright <2018> <Okinawa Institute of Science and Technology Graduate University>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

#import libraries
import numpy as np
import pandas as pd
import scipy

from IPython.core.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

import os
import imageio

from read_roi import read_roi_file
from read_roi import read_roi_zip

#Definition
#define the frames throughout the experiments
n_frames = 121
# define the time interval between each frame
t_inc = 5 # in minutes
print("Total frame of time lapse is %d" %(n_frames))
print("Time interval is %d minutes"%(t_inc))
#define the data location
location = r'D:\Cell electrotaxis\20180603similarity_analysis\metamorph\12phase_metamorphtracking.csv'
#define the data_type = 'ImageJ', 'LineageMapper', or 'Metamorph'
data_type = 'Metamorph'

#input data loading
if data_type=='ImageJ':
	df_ij = pd.read_csv(location)
	n_cells_ij = int(len(df_ij) / n_frames)
	timestamps = np.linspace(0, n_frames*t_inc, n_frames+1)
	print("Cell track numbers is %d"%(n_cells_ij))
elif data_type=='LineageMapper':
	df_LM = pd.read_csv(location)
	count = df_LM['Cell ID'].value_counts()
	cell_ids_LM = count[count==n_frames].index.tolist()
	n_cells_LM = int(len(cell_ids_LM))
	timestamps = np.linspace(0, n_frames*t_inc, n_frames+1)
	print("Cell track number is: " + str(n_cells_LM))
	col_names = df_LM.columns.tolist()
	selected_df = pd.DataFrame(columns=col_names)
	for i in cell_ids_LM:
		selected_df = selected_df.append(df_LM.loc[df_LM['Cell ID']==i].copy())
	selected_df.reset_index(drop=True, inplace=True)
elif data_type=='Metamorph':
	df_meta = pd.read_csv(location)
	count = df_meta['Object #'].value_counts()
	cell_ids_meta = count[count==n_frames].index.tolist()
	n_cells_meta = int(len(cell_ids_meta))
	timestamps = np.linspace(0, n_frames*t_inc, n_frames+1)
	print("Cell track number is:" + str(n_cells_meta))
	col_names = df_meta.columns.tolist()
	selected_df = pd.DataFrame(columns=col_names)
	for i in cell_ids_meta:
		selected_df = selected_df.append(df_meta.loc[df_meta['Object #']==i].copy())
	selected_df.reset_index(drop=True, inplace=True)
else:
	print("Data loading error")
#start processing data:
if data_type=='ImageJ':
	# Process the data into a numpy time-array
	props_t_array = []
	props_t_array = np.empty([n_cells_ij, 14, n_frames]) # Creates a time array, formatted like a spreadsheet, cells in rows, columns for X and Y, and t in Z
	#print(np.shape(props_t_array))
	cell_dfs = []
	ind_i = 0
	i_cell = 0
	for i in range(1,len(df_ij)): # Using 1 instead of zero here avoids indexing -1, but won't skip first row being copied because ind_i initialized as zero above.
		if(df_ij.loc[i-1,'Slice'] > df_ij.loc[i,'Slice']):
			ind_f = i - 1
			sub_df = df_ij.loc[ind_i:ind_f,:]
			ind_i = i
			# Copy the measurements of interest into the numpy array
			props_t_array[i_cell,0,:] = sub_df['X'] # This will be a problem if the number of frames ever differs between cells.
			props_t_array[i_cell,1,:] = sub_df['Y']
			props_t_array[i_cell,2,:] = sub_df['Area']
			props_t_array[i_cell,3,:] = sub_df['Perim.']
			props_t_array[i_cell,4,:] = sub_df['Angle']
			props_t_array[i_cell,5,:] = sub_df['Circ.']
			cell_dfs.append(sub_df) # add also to a list of dataframes
			i_cell = i_cell + 1

	if(i == len(df_ij) - 1): # A special case for the last cell in the results file.
		ind_f = i        
		sub_df =df_ij.loc[ind_i:ind_f,:]
		# Copy the measurements of interest into the numpy array
		props_t_array[i_cell,0,:] = sub_df['X'] # This will be a problem if the number of frames ever differs between cells.
		props_t_array[i_cell,1,:] = sub_df['Y']
		props_t_array[i_cell,2,:] = sub_df['Area']
		props_t_array[i_cell,3,:] = sub_df['Perim.']
		props_t_array[i_cell,4,:] = sub_df['Angle']
		props_t_array[i_cell,5,:] = sub_df['Circ.']
		# Correct the position coordinates so that all cells start at the same location in the plot.
	zerod_t_array = np.empty([n_cells_ij, 2, n_frames]) # Creates a time array, formatted like a spreadsheet, cells in rows, columns for X and Y, and t in Z

	for i in range(0,n_cells_ij):
		for j in range(0,n_frames):
			zerod_t_array[i,0,j] = props_t_array[i,0,j] - props_t_array[i,0,0]
			zerod_t_array[i,1,j] = props_t_array[i,1,j] - props_t_array[i,1,0]
	print(props_t_array.to_string())
	n_cells = n_cells_ij  
    
       
elif data_type=='LineageMapper':
	print("processing lineage mapper data")
	props_t_array = []
	props_t_array = np.empty([n_cells_LM, 14, n_frames])
	print(np.shape(props_t_array))
	n_rows_csv = len(selected_df)
	print('Number of cells: ' + str(n_cells_LM))
	print('Number of rows: ' + str(n_rows_csv))
	if(int(n_rows_csv / n_cells_LM) != n_frames): # We can use this to parse the file
		print('Error: improper number of rows in trk file for the number of cells and timepoints.') 
	cell_dfs = []
	ind_i = 0
	for i_cell in range(0,n_cells_LM):
		ind_f = ind_i + n_frames - 1
		sub_df = selected_df.loc[ind_i:ind_f,:]
		props_t_array[i_cell,0,:] = sub_df['X Coordinate']
		props_t_array[i_cell,1,:] = sub_df['Y Coordinate']
		# Display the current dataframe and portion of the numpy array.
		display(sub_df)
		print(props_t_array[i_cell,0:2,:])
		ind_i = ind_i + n_frames
	n_cells = n_cells_LM 
elif data_type=='Metamorph':
	print("processing metamorph data")
	props_t_array = []
	props_t_array = np.empty([n_cells_meta, 14, n_frames])
	print(np.shape(props_t_array))
	n_rows_csv = len(selected_df)
	print('Number of cells: ' + str(n_cells_meta))
	print('Number of rows: ' + str(n_rows_csv))
	if(int(n_rows_csv / n_cells_meta) != n_frames): # We can use this to parse the file
		print('Error: improper number of rows in the file for the number of cells and timepoints.') 
	cell_dfs = []
	ind_i = 0
	for i_cell in range(0,n_cells_meta):
		ind_f = ind_i + n_frames - 1
		sub_df = selected_df.loc[ind_i:ind_f,:]
		props_t_array[i_cell,0,:] = sub_df['X']
		props_t_array[i_cell,1,:] = sub_df['Y']
		# Display the current dataframe and portion of the numpy array.
		print(props_t_array[i_cell,0:2,:])
		ind_i = ind_i + n_frames
	n_cells = n_cells_meta
else:
	print("no data found")
#Calculation for cell migration parameters
if data_type=='ImageJ':
	for i in range(0,n_cells):
		for j in range(0, n_frames):
			#Segment length
			if(j > 0):
				segment = np.sqrt(pow((props_t_array[i,0,j]-props_t_array[i,0,j-1]),2) + pow((props_t_array[i,1,j]-props_t_array[i,1,j-1]),2)) 
			else:
				segment = 0    
			props_t_array[i,6,j] = segment
			# Cumulative path length 
			if(j > 0):
				cumulative = cumulative + segment
			else:
				cumulative = 0
			props_t_array[i,7,j] = cumulative
			# Orientation # CURRENTLY: If data_imageJ is false, then this is dealing with NaNs from the empty column of the array. 
			axis_angle = props_t_array[i,4,j]  # Angle of the long axis of the cell: Angle (IJ)??
			orientation = np.cos(2 * np.radians(axis_angle))
			props_t_array[i,8,j] = orientation       
			# Euclidean distance (From start to current frame)
			if(j > 0):
				euc_dist = np.sqrt(pow((props_t_array[i,0,j]-props_t_array[i,0,0]),2) + pow((props_t_array[i,1,j]-props_t_array[i,1,0]),2))
			else:
				euc_dist = 0
			props_t_array[i,9,j] = euc_dist
			# Migration speed
			if(j > 0):
				speed = segment / t_inc * 6 # Microns per hour, since t_inc is in minutes 
			else:
				speed = 0 # Or should it be NaN??
			props_t_array[i,10,j] = speed 
			# Directedness (Using the calculation from Paul's spreadsheet, where directedness = deltax / radius (euc_distance))
			if(j > 0): # Doesn't make sense to calculate this on the first frame.
				directedness = (props_t_array[i,0,j]-props_t_array[i,0,0]) / euc_dist
			else: 
				directedness = 0
			props_t_array[i,11,j] = directedness       
			# Turn angle
			if(j > 0): # Doesn't make sense to calculate this on the first frame.
				turn_angle_radians = np.arctan((props_t_array[i,1,j] - props_t_array[i,1,j-1]) / (props_t_array[i,0,j] - props_t_array[i,0,j-1]))
				turn_angle = np.degrees(turn_angle_radians)
			else: 
				turn_angle = 0
			props_t_array[i,12,j] = turn_angle
			# Endpoint directionality ratio (confinement ratio, meandering index)
			if(j > 0):
				ep_dr = cumulative / euc_dist # This is problematic because segment uses i+1 - i, whereas euc_dist uses i - 0.
			else: 
				ep_dr = 0
			#endpoint direcionality ratio is defined arbitrarily 0 at first frame 
			# Direction autocorrelation
			if(j > 0):
				dir_auto =  np.cos(props_t_array[i,12,j] - props_t_array[i,12,j-1])
			else: 
				dir_auto = 0
			props_t_array[i,13,j] = dir_auto
else:
	for i in range(0,n_cells):
		for j in range(0, n_frames):
			#Segment length
			if(j > 0):
				segment = np.sqrt(pow((props_t_array[i,0,j]-props_t_array[i,0,j-1]),2) + pow((props_t_array[i,1,j]-props_t_array[i,1,j-1]),2)) 
			else:
				segment = 0    
			props_t_array[i,6,j] = segment
			# Cumulative path length 
			if(j > 0):
				cumulative = cumulative + segment
			else:
				cumulative = 0
			props_t_array[i,7,j] = cumulative
			# Euclidean distance (From start to current frame)
			if(j > 0):
				euc_dist = np.sqrt(pow((props_t_array[i,0,j]-props_t_array[i,0,0]),2) + pow((props_t_array[i,1,j]-props_t_array[i,1,0]),2))
			else:
				euc_dist = 0
			props_t_array[i,9,j] = euc_dist
			# Migration speed
			if(j > 0):
				speed = segment / t_inc * 6 # Microns per hour, since t_inc is in minutes 
			else:
				speed = 0 # Or should it be NaN??
			props_t_array[i,10,j] = speed 
			# Directedness (Using the calculation from Paul's spreadsheet, where directedness = deltax / radius (euc_distance))
			if(j > 0): # Doesn't make sense to calculate this on the first frame.
				directedness = (props_t_array[i,0,j]-props_t_array[i,0,0]) / euc_dist
			else: 
				directedness = 0
			props_t_array[i,11,j] = directedness       
			# Turn angle
			if(j > 0): # Doesn't make sense to calculate this on the first frame.
				turn_angle_radians = np.arctan((props_t_array[i,1,j] - props_t_array[i,1,j-1]) / (props_t_array[i,0,j] - props_t_array[i,0,j-1]))
				turn_angle = np.degrees(turn_angle_radians)
			else: 
				turn_angle = 0
			props_t_array[i,12,j] = turn_angle
			# Endpoint directionality ratio (confinement ratio, meandering index)
			if(j > 0):
				ep_dr = cumulative / euc_dist # This is problematic because segment uses i+1 - i, whereas euc_dist uses i - 0.
			else: 
				ep_dr = 0
			#endpoint direcionality ratio is defined arbitrarily 0 at first frame 
			# Direction autocorrelation
			if(j > 0):
				dir_auto =  np.cos(props_t_array[i,12,j] - props_t_array[i,12,j-1])
			else: 
				dir_auto = 0
			props_t_array[i,13,j] = dir_auto

# Correct the position coordinates so that all cells start at the same location in the plot.
zerod_t_array = np.empty([n_cells, 2, n_frames]) # Creates a time array, formatted like a spreadsheet, cells in rows, columns for X and Y, and t in Z

for i in range(0,n_cells):
	for j in range(0,n_frames):
		zerod_t_array[i,0,j] = props_t_array[i,0,j] - props_t_array[i,0,0]
		zerod_t_array[i,1,j] = props_t_array[i,1,j] - props_t_array[i,1,0]

#export the descriptive statistics to a csv file
stats_df = pd.DataFrame(columns=['cell_id','time', 'x_pos_microns', 'y_pos_microns', 'x_pos_corr', 'y_pos_corr',
                                 'area', 'perimeter', 'angle', 'circularity', 'segment_length', 'cumulative_path_length', 
                                 'orientation', 'euclidean_distance', 'speed', 'directedness', 'turn_angle', 'direction_autocorrelation']) #deleted velocity
stats_df.round(4)
summary_cell_df = pd.DataFrame(columns=['cell_id', 'avg_area', 'avg_perimeter', 'avg_angle', 'avg_circularity', 'avg_segment_length', 'total_path_length', 
                                 'avg_orientation', 'euclidean_distance', 'avg_speed', 'avg_velocity', 'avg_directedness', 'avg_turn_angle', 'avg_direction_autocorrelation'])
summary_cell_df.round(2)
summary_timepoint_df = pd.DataFrame(columns=['time', 'avg_area', 'avg_perimeter', 'avg_angle', 'avg_circularity', 'avg_segment_length', 'total_path_length', 
                                 'avg_orientation', 'euclidean_distance', 'avg_speed', 'avg_velocity','avg_directedness', 'avg_turn_angle', 'avg_direction_autocorrelation'])
summary_timepoint_df.round(2)
t = np.linspace(0,(n_frames-1)*t_inc,n_frames)
i_row = 0
if data_type=='ImageJ':
	for i in range(0,len(props_t_array[:,0,0])):
		for j in range(0,len(props_t_array[0,0,:])):
			stats_df.loc[i_row] = i_row
			stats_df['cell_id'][i_row] = i + 1
			stats_df['time'][i_row] = t[j]
			stats_df['x_pos_microns'][i_row]  = props_t_array[i,0,j] 
			stats_df['y_pos_microns'][i_row] = props_t_array[i,1,j] 
			stats_df['x_pos_corr'][i_row] = zerod_t_array[i,0,j]
			stats_df['y_pos_corr'][i_row] = zerod_t_array[i,1,j]
			stats_df['area'][i_row] = props_t_array[i,2,j]
			stats_df['perimeter'][i_row] = props_t_array[i,3,j]
			stats_df['angle'][i_row] = props_t_array[i,4,j]
			stats_df['circularity'][i_row] = props_t_array[i,5,j]
			stats_df['segment_length'][i_row] = props_t_array[i,6,j]
			stats_df['cumulative_path_length'][i_row] = props_t_array[i,7,j]
			stats_df['orientation'][i_row] = props_t_array[i,8,j]
			stats_df['euclidean_distance'][i_row] = props_t_array[i,9,j]
			stats_df['speed'][i_row] = props_t_array[i,10,j]
			stats_df['directedness'][i_row] = props_t_array[i,11,j]   
			stats_df['turn_angle'][i_row] = props_t_array[i,12,j]
			stats_df['direction_autocorrelation'][i_row] = props_t_array[i,13,j]
			i_row = i_row + 1
else:
	for i in range(0,len(props_t_array[:,0,0])):
		for j in range(0,len(props_t_array[0,0,:])):
			stats_df.loc[i_row] = i_row
			stats_df['cell_id'][i_row] = i + 1
			stats_df['time'][i_row] = t[j]
			stats_df['x_pos_microns'][i_row]  = props_t_array[i,0,j] 
			stats_df['y_pos_microns'][i_row] = props_t_array[i,1,j] 
			stats_df['x_pos_corr'][i_row] = zerod_t_array[i,0,j]
			stats_df['y_pos_corr'][i_row] = zerod_t_array[i,1,j]
			stats_df['segment_length'][i_row] = props_t_array[i,6,j]
			stats_df['cumulative_path_length'][i_row] = props_t_array[i,7,j]
			stats_df['euclidean_distance'][i_row] = props_t_array[i,9,j]
			stats_df['speed'][i_row] = props_t_array[i,10,j]
			stats_df['directedness'][i_row] = props_t_array[i,11,j]   
			stats_df['turn_angle'][i_row] = props_t_array[i,12,j]
			stats_df['direction_autocorrelation'][i_row] = props_t_array[i,13,j]
			i_row = i_row + 1

#create avg statistics of individual cells
if data_type=='ImageJ':
	for i in range(0,len(props_t_array[:,0,0])):
		summary_cell_df.loc[i] = i
		summary_cell_df['cell_id'][i] = i + 1
		summary_cell_df['avg_area'][i] = np.mean(props_t_array[i,2,:])
		summary_cell_df['avg_perimeter'][i] = np.mean(props_t_array[i,3,:])
		summary_cell_df['avg_angle'][i] = np.mean(props_t_array[i,4,:])
		summary_cell_df['avg_circularity'][i] = np.mean(props_t_array[i,5,:])
		summary_cell_df['avg_segment_length'][i] = np.mean(props_t_array[i,6,1:]) # substract time point 0
		summary_cell_df['total_path_length'][i] = props_t_array[i,7,-1] # Total path length is cumulative path length at final timepoint
		summary_cell_df['avg_orientation'][i] = np.mean(props_t_array[i,8,:]) 
		summary_cell_df['euclidean_distance'][i] = props_t_array[i,9,-1] # Linear distance between first and last point
		summary_cell_df['avg_speed'][i] = np.mean(props_t_array[i,10,1:]) #subtract time point 0
		summary_cell_df['avg_velocity'][i] = props_t_array[i,9,-1] / ((t[-1] - t[0]) / 60) # Total Euclidean distance per hour. 
		summary_cell_df['avg_directedness'][i] = np.nanmean(props_t_array[i,11,1:])#subtract time point 0
		summary_cell_df['avg_turn_angle'][i] = np.nanmean(props_t_array[i,12,1:])#subtract time point 0
		summary_cell_df['avg_direction_autocorrelation'][i] = np.nanmean(props_t_array[i,13,1:])#subtract time point 0
else:
	for i in range(0,len(props_t_array[:,0,0])):
		summary_cell_df.loc[i] = i
		summary_cell_df['cell_id'][i] = i + 1
		summary_cell_df['avg_segment_length'][i] = np.mean(props_t_array[i,6,1:]) # substract time point 0
		summary_cell_df['total_path_length'][i] = props_t_array[i,7,-1] # Total path length is cumulative path length at final timepoint
		summary_cell_df['euclidean_distance'][i] = props_t_array[i,9,-1] # Linear distance between first and last point
		summary_cell_df['avg_speed'][i] = np.mean(props_t_array[i,10,1:]) #subtract time point 0
		summary_cell_df['avg_velocity'][i] = props_t_array[i,9,-1] / ((t[-1] - t[0]) / 60) # Total Euclidean distance per hour. 
		summary_cell_df['avg_directedness'][i] = np.nanmean(props_t_array[i,11,1:])#subtract time point 0
		summary_cell_df['avg_turn_angle'][i] = np.nanmean(props_t_array[i,12,1:])#subtract time point 0
		summary_cell_df['avg_direction_autocorrelation'][i] = np.nanmean(props_t_array[i,13,1:])#subtract time point 0	



#individual time point statistics
if data_type=='ImageJ':
	for i in range(0,len(props_t_array[0,0,:])):
		summary_timepoint_df.loc[i] = i
		summary_timepoint_df['time'][i] = i*t_inc
		summary_timepoint_df['avg_area'][i] = np.mean(props_t_array[:,2,i])
		summary_timepoint_df['avg_perimeter'][i] = np.mean(props_t_array[:,3,i])
		summary_timepoint_df['avg_angle'][i] = np.mean(props_t_array[:,4,i])
		summary_timepoint_df['avg_circularity'][i] = np.mean(props_t_array[:,5,i])
		summary_timepoint_df['avg_segment_length'][i] = np.mean(props_t_array[:,6,i])
		summary_timepoint_df['total_path_length'][i] = np.mean(props_t_array[:,7,i]) # Total path length is cumulative path length at final timepoint
		summary_timepoint_df['avg_orientation'][i] = np.mean(props_t_array[:,8,i])
		summary_timepoint_df['euclidean_distance'][i] = np.mean(props_t_array[:,9,i]) # Linear distance between first and last point
		summary_timepoint_df['avg_speed'][i] = np.mean(props_t_array[:,10,i])
		summary_timepoint_df['avg_velocity'][i] = np.mean(props_t_array[:,9,i]) / ((t[-1] - t[0]) / 60) # Total Euclidean distance per hour. 
		summary_timepoint_df['avg_directedness'][i] = np.nanmean(props_t_array[:,11,i])   
		summary_timepoint_df['avg_turn_angle'][i] = np.nanmean(props_t_array[:,12,i])
		summary_timepoint_df['avg_direction_autocorrelation'][i] = np.nanmean(props_t_array[:,13,i])
else:
	for i in range(0,len(props_t_array[0,0,:])):
		summary_timepoint_df.loc[i] = i
		summary_timepoint_df['time'][i] = i*t_inc
		summary_timepoint_df['avg_segment_length'][i] = np.mean(props_t_array[:,6,i])
		summary_timepoint_df['total_path_length'][i] = np.mean(props_t_array[:,7,i]) # Total path length is cumulative path length at final timepoint
		summary_timepoint_df['euclidean_distance'][i] = np.mean(props_t_array[:,9,i]) # Linear distance between first and last point
		summary_timepoint_df['avg_speed'][i] = np.mean(props_t_array[:,10,i])
		summary_timepoint_df['avg_velocity'][i] = np.mean(props_t_array[:,9,i]) / ((t[-1] - t[0]) / 60) # Total Euclidean distance per hour. 
		summary_timepoint_df['avg_directedness'][i] = np.nanmean(props_t_array[:,11,i])   
		summary_timepoint_df['avg_turn_angle'][i] = np.nanmean(props_t_array[:,12,i])
		summary_timepoint_df['avg_direction_autocorrelation'][i] = np.nanmean(props_t_array[:,13,i])

export_path = 'export//spreadsheets//'
if not os.path.exists(export_path):
	os.makedirs(export_path)
stats_df.to_csv(export_path + "cell_migration_descriptive_statistics.csv", header=True, index=False)
summary_cell_df.to_csv(export_path + "cell_migration_summary.csv", header=True, index=False)
summary_timepoint_df.to_csv(export_path+"timepoint_migration_summary.csv", header=True, index=False)


#Drawing plots
# Set plot limits
xmin = -500
xmax = 500
ymin = -500
ymax = 500

frames = []
fig = plt.figure(frameon=True,facecolor='w')
fig.set_size_inches(10,10)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)

export_path = 'export//scatter//'
if not os.path.exists(export_path):
    os.makedirs(export_path)

frames = []
fig = plt.figure(frameon=True,facecolor='w')
fig.set_size_inches(10,10)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
cell_colors = np.linspace(0,1,n_cells)


    
for t in range(0,n_frames):  
    ax=plt.subplot(111)
    ax.clear()
    ax.scatter(props_t_array[:,0,t],props_t_array[:,1,t],s=20,  alpha=1, c=cell_colors)
    #ax.axis('equal')
    ax.axis([xmin, xmax, ymin, ymax]) # Setting the axes like this avoid the zero values in the preallocated empty array.
    #ax.text(250, 1050, 'Distribution of cell positions at t = ' + str(int(timestamps[t])) + ' minutes', fontsize=15)
    ax.text(xmax / 6, ymax + 10, 'Distribution of cell positions at t = ' + str(int(timestamps[t])) + ' minutes', fontsize=15)
    ax.set_xlabel('X position ($\mu$m)', fontsize=15)
    ax.set_ylabel('Y position ($\mu$m)', fontsize=15)    
    # Draw the figure
    fig.canvas.draw()

    # Convert to numpy array, and append to list
    np_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(np_fig)
    
imageio.mimsave(export_path + '/scatter_raw.gif', frames)
np.fig=[]
frames = []
fig = plt.figure(frameon=True,facecolor='w')
fig.set_size_inches(10,10)

for t in range(0,n_frames):
    ax = plt.subplot(111)
    ax.clear()
    ax.scatter(zerod_t_array[:,0,t],zerod_t_array[:,1,t],s=20,  alpha=1, c=cell_colors)
    #ax.axis('equal')
    #plt.axis('off')
    ax.axis([xmin, xmax, ymin, ymax]) # Setting the axes like this avoid the zero values in the preallocated empty array.
    #ax.text(-50, 90, 'Distribution of cell positions (zeroed) at t = ' + str(int(timestamps[t])) + ' minutes', fontsize=15)
    ax.text(xmin + (xmax - xmin) / 8, ymax + 5, 'Distribution of cell positions (zeroed) at t = ' + str(int(timestamps[t])) + ' minutes', fontsize=15)
    ax.set_xlabel('Relative X position ($\mu$m)', fontsize=15)
    ax.set_ylabel('Relative Y position ($\mu$m)', fontsize=15)

    # Draw the figure
    fig.canvas.draw()
    #uncomment this one if you want to save individual time point into a file
    #plt.savefig(export_path + 'scatter%d.png'%(t), format='png', dpi=600)
    # Convert to numpy array, and append to list
    np_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(np_fig)
    
imageio.mimsave(export_path + '/scatter_zeroed.gif', frames)

#2D hair ball plots with each cell track is one color
x = zerod_t_array[:,0,:]
y = zerod_t_array[:,1,:]
t = np.linspace(0,n_frames*t_inc,n_frames)
fig = plt.figure(figsize = (10,10),facecolor='w')
ax = fig.add_subplot(111)

export_path = 'export//2d_hairball//'
if not os.path.exists(export_path):
    os.makedirs(export_path)
    
segs = np.zeros((n_cells, n_frames, 2), float)   
segs[:, :, 0] = x
segs[:, :, 1] = y


ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_xlabel('X position ($\mu$m)')
ax.set_ylabel('Y position ($\mu$m)')

colors = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

line_segments = LineCollection(segs,colors=colors, cmap=plt.get_cmap('jet'))

ax.add_collection(line_segments)
ax.set_title('Cell migration trajectories')
#plt.axis('equal')
plt.savefig(export_path + '2d_hairball.png', format='png', dpi=600)

#2D hair ball trajectory plot with color coed accroding to elapsed time (Imaris Like trajectory)
t = np.linspace(0,n_frames*t_inc,n_frames)

fig = plt.figure(figsize = (10,10),facecolor='w')
ax = fig.add_subplot(111)

export_path = 'export//2d_hairball//'
if not os.path.exists(export_path):
    os.makedirs(export_path)
    
    
for n in range(0,n_cells):

    x = zerod_t_array[n,0,:]
    y = zerod_t_array[n,1,:]
    
    # Remove the nans from the array
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    # Set the segments in the correct format
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Axis limits and titles
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)    
    ax.set_xlabel('X position ($\mu$m)')
    ax.set_ylabel('Y position ($\mu$m)')
#     plt.axis('equal')
    # Set the colormap
    cmap=plt.get_cmap('jet')
    line_segments = LineCollection(segments,array=t, cmap=cmap)
    ax.add_collection(line_segments)
    
axcb = fig.colorbar(line_segments)
axcb.set_label('Time (minutes)')
ax.set_title('Cell migration trajectories')
#plt.axis('equal')
 
plt.savefig(export_path + '2d_hairball_time_cmap.png', format='png', dpi=600)

# 2D hairball with color of entire trjactory is mapped by color
x = zerod_t_array[:,0,:]
y = zerod_t_array[:,1,:]
end_x_pos = np.empty([len(x[:,0]),1])

for i in range(0,len(end_x_pos)): # For each cell
    x_vals = np.copy(np.squeeze(x[i,:]))
    x_vals = x_vals[~np.isnan(x_vals)]

    if(len(x_vals) > 0):
        end_x_pos[i] = x_vals[-1]
fig = plt.figure(figsize = (10,10),facecolor='w')
ax = fig.add_subplot(111)

export_path = 'export//2d_hairball//'
if not os.path.exists(export_path):
    os.makedirs(export_path)

segs = np.zeros((n_cells, n_frames, 2), float)   
segs[:, :, 0] = x
segs[:, :, 1] = y

#ax.set_xlim(np.nanmin(x), np.nanmax(x))
#ax.set_ylim(np.nanmin(y), np.nanmax(y))
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_xlabel('X position ($\mu$m)')
ax.set_ylabel('Y position ($\mu$m)')

line_segments = LineCollection(segs,array=np.squeeze(end_x_pos), cmap=plt.get_cmap('jet'))
ax.add_collection(line_segments)
ax.set_title('Cell migration trajectories')
axcb = fig.colorbar(line_segments, orientation="horizontal", pad=0.1)
axcb.set_label('Final x position ($\mu$m)')

plt.savefig(export_path + '2d_hairball_cmap_endPos.png', format='png', dpi=600)

#2d hair ball plot with final position color coding by X direction (Ibidi like)
x = zerod_t_array[:,0,:]
y = zerod_t_array[:,1,:]
x_1 = x[x[:,-1] < 0]
y_1 = y[x[:,-1] < 0]
x_2 = x[x[:,-1] > 0]
y_2 = y[x[:,-1] > 0]


t = np.linspace(0,(n_frames-1)*t_inc,n_frames)

fig = plt.figure(figsize = (10,10),facecolor='w')
ax = fig.add_subplot(111)

export_path = 'export//2d_hairball//'
if not os.path.exists(export_path):
    os.makedirs(export_path)

segs_1 = np.zeros((len(x_1[:,0]), n_frames, 2), float)   
segs_1[:, :, 0] = x_1
segs_1[:, :, 1] = y_1

segs_2 = np.zeros((len(x_2[:,0]), n_frames, 2), float)   
segs_2[:, :, 0] = x_2
segs_2[:, :, 1] = y_2

#ax.set_xlim(np.nanmin(x)-5, np.nanmax(x)+5)
#ax.set_ylim(np.nanmin(y)-5, np.nanmax(y)+5)
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)    
ax.set_xlabel('X position ($\mu$m)')
ax.set_ylabel('Y position ($\mu$m)')

line_segments_1 = LineCollection(segs_1,colors='red')
ax.add_collection(line_segments_1)
line_segments_2 = LineCollection(segs_2,colors='black')
ax.add_collection(line_segments_2)
ax.set_title('Cell migration trajectories')
ax.scatter(x_1[:,-1],y_1[:,-1],s=50, c='red')
ax.scatter(x_2[:,-1],y_2[:,-1],s=50, c='black')

plt.savefig(export_path + '2d_hairball_cmap_endPos_2color.png', format='png', dpi=600)

#3D hairball plot with time as z axis
frames = []
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')

export_path = 'export//3d_hairball//'
if not os.path.exists(export_path):
    os.makedirs(export_path)
       
for n in range(0,n_cells):

    x = zerod_t_array[n,0,:]
    y = zerod_t_array[n,1,:]
    t = np.linspace(0,n_frames*t_inc,n_frames)

    ax.plot(x, y, t)#, c=)
    ax.set_xlabel('X position ($\mu$m)')
    ax.set_ylabel('Y position ($\mu$m)')
    ax.set_zlabel('Time (minutes)')
    

for angle in range(0, 360):
    ax.view_init(30, angle)
    
    # Draw the figure
    fig.canvas.draw()

    # Convert to numpy array, and append to list
    np_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(np_fig)
    
imageio.mimsave(export_path + '/3d_hairball_test.gif', frames)

#plot violin plot for cell area perimeter, orientation, circularity, speed, directedness, turn angle, and direction autocorrelation
export_path = 'export//Violin plots//'
if not os.path.exists(export_path):
    os.makedirs(export_path)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
t=np.linspace(0,(n_frames-1)*t_inc, n_frames)
ax.violinplot(np.squeeze(props_t_array[:,2,:]), positions=t, widths=10, showmeans=False, showextrema=True, showmedians=True)
ax.set_title('cell area')
ax.set_ylabel('cellarea($\mu m^2$)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'area_violin.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,3,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Cell perimeter')
ax.set_ylabel('Cell Perimeter ($\mu$m)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'perimeter_violin.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,4,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Orientation angle')
ax.set_ylabel('Angle (degrees)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'angle_violin.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,5,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Circularity')
ax.set_ylabel('Circularity (a.u.)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'circularity_violin.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,10,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Speed')
ax.set_ylabel('Speed ($\mu$m/h)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'speed_violin.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,11,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Directedness')
ax.set_ylabel('Directedness(a.u.)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'directedness_violin.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,12,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Turn angle')
ax.set_ylabel('Turn angle (degrees)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'turnangle_violin.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax.violinplot(np.squeeze(props_t_array[:,13,:]),positions=t, widths = 10, showmeans=True, showextrema=True, showmedians=False)
ax.set_title('Direction autocorrelation')
ax.set_ylabel('Direction autocorrelation(a.u.)')
ax.set_xlabel('Time (minutes)')
plt.savefig(export_path + 'direction_autocorrelation_violin.png', format='png', dpi=600)

#box plots
export_path = 'export//boxplots//'

if not os.path.exists(export_path):
    os.makedirs(export_path)

t = np.linspace(0,(n_frames-1)*t_inc,n_frames)    

linewidth = 1
fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,2,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,2,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Cell area')
ax.set_ylabel('Cell area ($\mu m^2$)')
ax.set_xlabel('Time (frame)')
plt.savefig(export_path + 'cellarea_boxplot.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,3,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,3,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Cell perimeter')
ax.set_ylabel('Cell perimeter ($\mu m$)')
ax.set_xlabel('Time (frame)')
plt.savefig(export_path + 'perimeter_boxplot.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,4,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,4,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Orientation angle')
ax.set_ylabel('Orientation angle (degrees)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'angle_boxplot.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,5,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,5,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Circularity')
ax.set_ylabel('Circularity (a.u.)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'circularity_boxplot.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,10,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,10,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Speed')
ax.set_ylabel('Speed ($\mu$m/h)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'speed_boxplot.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,11,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,11,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Directedness')
ax.set_ylabel('Directedness(a.u.)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'directedness_boxplot.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,12,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,12,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Turn angle')
ax.set_ylabel('Turn angle (degrees)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'turnangle_boxplot.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.boxplot(data=np.squeeze(props_t_array[:,13,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
ax = sns.swarmplot(data=np.squeeze(props_t_array[:,13,:]), orient="v", linewidth=linewidth, ax=ax)
ax.set_title('Direction autocorrelation')
ax.set_ylabel('Direction autocorrelation(a.u.)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'direction_autocorrelation_boxplot.png', format='png', dpi=600)

export_path = 'export//Timeseries plots//'

if not os.path.exists(export_path):
    os.makedirs(export_path)

    
    
t = np.linspace(0,(n_frames-1)*t_inc,n_frames)    

linewidth = 1
fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax=sns.tsplot(np.squeeze(props_t_array[:,2,:]),time=t, condition='Cell Area', value='Cell Area ($\mu m^2$)', err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Cell area')
ax.set_ylabel('Cell area ($\mu m^2$)')
ax.set_xlabel('Time (Minutes)')

plt.savefig(export_path + 'cellarea_timeseries.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax =sns.tsplot(np.squeeze(props_t_array[:,3,:]),time=t, condition='Cell Perimeter', value='Cell Perimeter ($\mu$m)',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Cell perimeter')
ax.set_ylabel('Cell perimeter ($\mu m$)')
ax.set_xlabel('Time (Minutes)')

plt.savefig(export_path + 'perimeter_timeseries.png', format='png', dpi=600)



fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.tsplot(np.squeeze(props_t_array[:,4,:]),time=t, condition='Orientation angle', value='Orientation Angle (degrees)',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Orientation angle')
ax.set_ylabel('Orientation angle (degrees)')
ax.set_xlabel('Time (Minutes)')

plt.savefig(export_path + 'angle_timeseries.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.tsplot(np.squeeze(props_t_array[:,5,:]),time=t, condition='Circularity', value='Circularity (a.u.)',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Circularity')
ax.set_ylabel('Circularity (a.u.)')
ax.set_xlabel('Time (Minutes)')

plt.savefig(export_path + 'circularity_timeseries.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.tsplot(np.squeeze(props_t_array[:,10,1:-1]),time=t[1:-1], condition='Speed', value='Speed ($\mu$m)/h',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Speed')
ax.set_ylabel('Speed ($\mu$m/h)')
ax.set_xlabel('Time (Minutes))')

plt.savefig(export_path + 'speed_timeseries.png', format='png', dpi=600)

fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
ax = sns.tsplot(np.squeeze(props_t_array[:,11,1:-1]),time=t[1:-1], condition='Directedness', value='Directedness (a.u.)',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Directedness')
ax.set_ylabel('Directedness(a.u.)')
ax.set_xlabel('Time (Minutes)')

plt.savefig(export_path + 'directedness_timeseries.png', format='png', dpi=600)


fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
#ax = sns.boxplot(data=np.squeeze(props_t_array[:,12,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
#ax = sns.swarmplot(data=np.squeeze(props_t_array[:,12,:]), orient="v", linewidth=linewidth, ax=ax)
ax = sns.tsplot(np.squeeze(props_t_array[:,12,1:-1]),time=t[1:-1], condition='Turn angle', value='Turn angle (degree)',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Turn angle')
ax.set_ylabel('Turn angle (degrees)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'turnangle_timeseris.png', format='png', dpi=600)
plt.show()
fig = plt.figure(figsize=(18,10), facecolor='w')
ax = fig.add_subplot(111)
#ax = sns.boxplot(data=np.squeeze(props_t_array[:,13,:]),orient="v",linewidth=linewidth,fliersize=2, ax=ax)
#ax = sns.swarmplot(data=np.squeeze(props_t_array[:,13,:]), orient="v", linewidth=linewidth, ax=ax)
sns.tsplot(np.squeeze(props_t_array[:,13,1:-1]),time=t[1:-1], condition='Direction autocorrelation', value='Direction autocorrelation (a.u.)',
              err_style="ci_band", ci=[0,95], ax=ax)
ax.set_title('Direction autocorrelation')
ax.set_ylabel('Direction autocorrelation(a.u.)')
ax.set_xlabel('Time (frame)')

plt.savefig(export_path + 'direction_autocorrelation_boxplot.png', format='png', dpi=600)


export_path = 'export//frequency_histogram_subplots//'
if not os.path.exists(export_path):
    os.makedirs(export_path)
    

sns.set_style({'lines.linewidth': 8.0},{'axes.linewidth': 2.0})
frames = []
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,  figsize=(20,5), facecolor='w')

for t in range(0,n_frames):
   
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    #speed
    ax1.hist(props_t_array[:,10,t], color='white',edgecolor='black', linewidth=4,range=(0, np.nanmax(props_t_array[:,10,:])))
    ax1.set_xlabel('Speed ($\mu$m/h)', fontsize=20)
    ax1.set_ylabel('Frequency', fontsize=20)
    ax1.set_xlim(0, np.nanmax(props_t_array[:,10,:]))
    ax1.set_ylim(0, n_cells)
    
    #angle
    ax2.hist(props_t_array[:,4,t], color='white',edgecolor='black', linewidth=4,range=(0, 180)) #props_t_array[:,8,t])
    ax2.set_xlabel('$\Phi$ (degrees)', fontsize=20)
    ax2.set_ylabel('Frequency', fontsize=20)
    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, n_cells)
    #turn angle
    ax3.hist(props_t_array[:,12,t], color='white',edgecolor='black', linewidth=4,range=(np.nanmin(props_t_array[:,12,:]), np.nanmax(props_t_array[:,12,:])))
    ax3.set_xlabel('α(degrees)', fontsize=20)
    ax3.set_ylabel('Frequency', fontsize=20)
    ax3.set_xlim(np.nanmin(props_t_array[:,12,:]), np.nanmax(props_t_array[:,12,:]))
    ax3.set_ylim(0, n_cells)
    # plot of 2 variables
    p1=sns.kdeplot(zerod_t_array[:,0,t], shade=False, color="b", ax=ax4)
    p1=sns.kdeplot(zerod_t_array[:,1,t], shade=False, color="r", ax=ax4)



    ax4.text(-150, 0.06, '$\Delta$X', fontsize=30, color='blue')
    ax4.text(-150, 0.05, '$\Delta$Y', fontsize=30, color='red')
    ax4.set_xlabel('Change in position ($\mu$m)', fontsize=20)
    ax4.set_xlim(np.nanmin(zerod_t_array[:,:,:])-5, np.nanmax(zerod_t_array[:,:,:])+5)
    ax4.set_ylim(0, 0.1)
    
    # Draw the figure
    fig.canvas.draw()

    # Convert to numpy array, and append to list
    np_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(np_fig)
    
imageio.mimsave(export_path + '/frequency_histogram_subplots.gif', frames)
