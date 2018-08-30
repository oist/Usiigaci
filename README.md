# Usiigaci: Label-free instance-aware cell tracking under phase contrast microscopy using Mask R-CNN
**Hsieh-Fu Tsai<sup>1,2</sup>, Joanna Gajda<sup>3</sup>, Tyler F.W. Sloan<sup>4</sup>, Andrei Rares<sup>5</sup>, and Amy Q. Shen<sup>1</sup>**

<sup><sup>1</sup>Micro/Bio/Nanofluidics Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa Japan
<sup>2</sup>Research Fellow of Japan Society of Promotion for Science
<sup>3</sup>AGH University of Science and Technology, Krakow, Poland
<sup>4</sup>Quorumetrix Solutions, Canada
<sup>5</sup>ImagineA, The Netherlands</sup>
	
![T98G microscopy](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gelectrotaxis-1.gif)
![T98G results from Usiigaci](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gmask-3.gif)

[Usiigaci, ウシーガチ](http://ryukyu-lang.lib.u-ryukyu.ac.jp/srnh/details.php?ID=SN03227) in Ryukyuan language means "tracing", "透き写し" in Japanese，*i.e.* drawing the outline of objects based on a template. The process is essentially what we do: following the morphology and position of cells under microscope, analyze how cell respond upon environmental perturbation in the microenvironment. However, this process is bloody tedious if done by human, and now we developed a pipeline using the famous Mask R-CNN to do this for us. Letting us not only track objects by their position but also track how their morphology changes through time. 

Zernike's phase contrast microscopy is a brightfield microscopy technique developed by Frits Zernike and by inventing the phase contrast technique, he won the 1953 Nobel Prize for physics. Phase contrast microscopy is favored by biologists because it translates the phase difference caused by cell components into amplitude thus making these transparent structures more visible. Also, in comparison to differential interference contrast microscopy, phase contrast microscopy works without problems with different substrates especially on plastics that are highly birefringent. 

Phase contrast microscopy images are notoriously difficult to segment by conventional computer vision methods. However, accurate whole cell outline segmentation and resolution of cells that contact each other are essential as the first step for cell tracking in automated microscopy needs accurate cell identification. Tracking and visualization of the cellular dynamics based on the segmentations help us understand and quantitative analyze cellular dynamics. 

We report Usiigaci, a semi-automated pipeline to segment, track, and visualize cell migration in phase contrast microscopy.

High accuracy label-free instance-aware segmentation is achieved by adapting the mask regional convolutional neural network (Mask R-CNN), winner of Marr prize at ICCV 2017 by He *et al.*. We built Usiigaci's segmentation module based on the Mask R-CNN implementation by [Matterport](https://github.com/matterport/Mask_RCNN). Using 50 manually-annotated cell images for training, the trained Mask R-CNN neural network can generate high accuracy whole cell segmentation masks that allow us to analyze both cell migration and cell morphology which are difficult even by fluorescent imaging. 

Cell tracking and data verification can be done in ImageJ, other existin tracking software such as [Lineage Mapper](https://github.com/usnistgov/Lineage-Mapper), or Usiigaci tracker that we developed based on open-source [trackpy](https://soft-matter.github.io/trackpy/v0.3.2/) library. A GUI is also developed to allow manual data verification to check tracking results and delete bad results.  

A Jupyter Notebook and the corresponding python script are developed for automated processing and visualization of the tracked results. Step-centric and cell-centric parameters are automatically computed and saved into additional spreadsheets where users can access and reuse in statistical software or R. Automated visualization of cell migration is also generated for cell trajectory graphs, box plots, etc. 

* Cell trajectory graph
	* 2D hair ball color coded by track
	* 2D hair ball color coded by time (Imaris like)
	* 2D hair ball color coded by direction (Ibidi like)
	* 2D hair ball color coded by direction length
	* 3D hair ball with z as time 
	* scatter plot in gif
* Automated cell migration analysis
	* computation of step centric parameters
		* instantaneous displacement
		* instantaneous speed
		* turn angle
		* direction autocorrelation
		* Directedness
	* compuatation of cell centric parameters
		* cumulative distance (total traveled distance)
		* Euclidean distance
		* net velocity
		* end point directionality ratio
		* orientation (cell alignment index)
	* save individual cell track data
	* save summary of each cell throughout experiment
	* save summary of ensemble at each time point
* automated plotting of descriptive statistics
	* rose histogram of cell orientation
	* box plots, violin plots, and time series plots of cell migration parameters
	* frequency histograms

We worked on Usiigaci for our work on cell electrotaxis study, and hopefully can devote to current international effort to standardize cell migration experiments. 

We hope Usiigaci is interesting to you and if it is useful for your research, please cite the following paper.
```
Hsieh-Fu Tsai, Joanna Gajda, Tyler F.W. Sloan, Andrei Rares, and Amy Q. Shen, softwareX, inprep
```


Usiigaci is released under MIT License. 
```
Copyright (c) 2018 Okinawa Institute of Science & Technology Graduate University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the 
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions 
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
USE OR OTHER DEALINGS IN THE SOFTWARE.

```

TensorFlow is open-sourced under Apache 2.0 opensource license. 

Keras is released under MIT license

The copyright of PyQT belong to Riverbank Computing ltd.

Pandas is released under [BSD 3-Clause License](http://pandas.pydata.org/pandas-docs/stable/overview.html?highlight=bsd). Copyright owned by AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team. 

Trackpy is released under [BSD 3-Clause License](https://github.com/soft-matter/trackpy/blob/master/LICENSE). Copyright owned by trackpy contributors.

NumPy and SciPy are released under [BSD-new License]

Scikit-image is released under [modified BSD license](https://github.com/scikit-image/scikit-image)

PIMS is released under [modified BSD license](https://github.com/soft-matter/pims/blob/master/license.txt)

Matplotlib is released under [Python Software Foundation (PDF) license](https://matplotlib.org/)

Seaborn is released under [BSD 3-clause license](https://github.com/mwaskom/seaborn/blob/master/LICENSE)

ffmpeg is licensed under [LGPL 2.1 license](https://www.ffmpeg.org/legal.html)



## Future work
- [ ] pretrain model weights for DIC microscopy.
- [ ] Multiclass segmentation to realize identification of mitotic cells.
- [ ] Multiclass segmentation to realize label-free co-cultured cell segmentation.

## Acknowledgement:
This work is supported by JSPS KAKENHI Grant JP1700362 and Okinawa Institute of Science and Technology Graduate University with subsidy funding from the Cabinet Office, Government of Japan. 


## Dependencies
### Hardware
A computer with CUDA-ready GPU should be able to run Usiigaci. 
We built the development machine in linux environment on an Alienware 15 with NVIDIA GTX 1070 8GB GPU.
But Usiigaci has been verified working on the following machines.
1. Windows 10 64 bit on Alienware 15 with GTX 1070 8GB
2. Linux 64 bit on Alienware 15 with GTX1070 8GB or with a GTX 1080Ti 11GB in an Alienware graphics amplifier
3. Windows 7 64 bit on Dell Precision Workstation T7810 with Quadro M4000

We see a high efficiency on running Usiigaci in a linux machine.

### Mask R-CNN
#### our working linux setup
* Kubuntu 16.04 Linux
* NVIDIA graphics card with compute capability > 5.0
* CUDA 9.1
* TensorFlow 1.4
* Keras 2.1.2
* Anaconda with Python 3.6 

#### our working windows setup
* Windows 10 64bit
* NVIDIA graphics card GTX 1070 8GB
* Anaconda with Python 3.6
* Tensorflow 1.9 with GPU
* CUDA 9.0
* CuDNN 7.1.4 for CUDA9.0
* Keras 2.1.6 

The exact version of CUDA and Keras is required to work with the Matterport Mask R-CNN repo.


### Python tracking GUI
* Python 3.4+
* Trackpy
* Scipy
* scikit-image
* Numpy
* Pandas
* Matplotlib
* PIMS
* PyQt5
* PyQtGraph
* ffmpeg
* Pandas


### Single cell migration data analysis Notebook
* Python3.4+
* Numpy
* Scipy
* Pandas
* Matplotlib
* seaborn
* imageio
* read_roi
* (jupyter-navbar)

## How to use Usiigaci 
### Video tutorial 
[Youtube link](https://www.youtube.com/watch?v=nH2oc-o5K10)

### Segmentation using Mask R-CNN

1. Download the trained weights
	
	Due to the file size limit of Github, please download the three weights we used for our phase contrast microscope in the [Dropbox folder](https://www.dropbox.com/sh/3eldgvytfchm9gr/AAB6vzPaEf8buk81IRVNClUEa?dl=0).

	Note: These are trained for phase contrast images on Nikon Ti-E with 10X phase contrast objective, 1.5X intermediate magnification with a Hamamatsu Orca Flash V4.0 sCMOS camera at 1024x1022 size. We have found Mask R-CNN to be more resilient to environmental changes, but if the results from pretrained weights are suboptimal, you can see the last section to train the network with your own data.




The inference script "/Mask R-CNN/Inference.py" is the script you need to run on images for generating corresponding masks. 
1. (organize you image data)

	assuming you're using NIS element, you can export the images by xy, t, and c if you have one.

	use the NIS_export_organize sorting script to the images so that for each xy dimension, all the time step images are organized in a folder.

2. edit the inference.py script

	line 288 change the path to the folder you want to run inference. 

	The script will automatically search all th nested directories of this folder and run inference on each of them.

	line 292:294 change the path to the trained weights. 

	you can specify multiple weights from different training, the inference code will run predictions using each model weight and average them. This costs more time to do inference. But since all neural networks can have false negatives (blinking), this can alleviate the false result frequency.

	line 296 adjust the model_list by the model_path you define.


3. run the inference.py script.

	```
	python inference.py
	```
	the keras and tensorflow should take care of looking for the main GPU and run the inference. 

4. instance-aware mask images will saved in a mask folder be right next to the folder you defined and you can use it to run in Lineage Mapper or etc.

### Data verification/tracking

#### Tracking on ImageJ

1. Load the images in ImageJ as a stack, you can verify the data and do manual tracking.
2. for ease of tracking, you can use an ImageJ plugin developed by [Emanuele Martini](https://bitbucket.org/e_martini/fiji_plugins/overview). 
	1. threshold the instance-aware stack to binary masks.
	2. run the plugin, you can find one target cells on the first slice with magic wand tool and click ok, based on overlapping, the plugin will find the target ROIs in the rest of the slices and add them into ROI manager.
	3. in ROI manager, you can edit each ROI and click "Measure" to output measured results for further analysis.

#### Tracking using Usiigaci tracker

A python tracking software is developed based on the [trackpy](https://soft-matter.github.io/trackpy/v0.3.2/) libraries and others. 
The tracker is the work by Dr. Andrei Rares. 

##### Prerequisite:
1. modify the Imageitem class of PyQtGraph
overwrite the ImageItem.py into python/site-packages/pyqtgraph/graphicsItems folder


##### Using the Usiigaci tracker:

![Usiigaci tracker GUI](https://github.com/oist/Usiigaci/blob/master/Demo/GUI.png)

1. Launch the tracker GUI by 

	```
	python cell_main.py
	```
	The left panel is for displaying raw image. Right panel is for displaying mask images. Bottom are two synced scrollbars so users can check each frame of a time lapse experiments.
	On the top left is the parameters setting. If the mask folder bear a different suffix, please change here. On the right is a list for all the cell tracks.
	Upon finish of tracking, numbered ids and cell track list will be updated. 


2. open the folder to the cell microscopy images. The tracker will load the segmented masks by looking at mask folder name with maskfolder usffix
3. Run cell tracking and the cell tracking will be done on using the mask generated from Mask R-CNN. 

	The tracking results that are suboptimal from segmentation error were repaired. 
	The tracking takes about a minute or two depending on the number of objects and frames.

4. After tracking is finished. If there are bad tracks. Users can deselect that track in the cell list.

	Alternatively, users can also take advantage of select all tracks, or select complete tracks (only the tracks that are recognized and tracked through all the frames).

5. Click "Save selection" to save the track results into "tracks.csv", labeled images, as well as rendered movies (you need ffmpeg) into a folder.

6. The XY coordinate, area, perimeter, and angle are extracted using scikit-image's measure.regionprops. 
	Note: angle between long axis and x axis in radians was extracted using region.orientation, but not to confuse with the orientation index in cell migration, which is cosine 2\*angle.


#### Tracking using other tracking software

Alternatively, you can load the indexed 8 bit masks files into Lineage Mapper, or Metamorph which the tracking can be easily done.

### Data analysis and visualization
Currently we have coded data loading interfaces for 5 types of analyzed data. but you can look into the codes to adapt the output data in your preference also.
One needs to make sure the data outputed from each tracking software is already calibrated into micron since the data analysis script does not take account of this anymore.
1. ImageJ tracked multi-measure output 

	each cell track of all time points followed by another

2. Lineage Mapper tracked results

	only cells that did not divide, fusion or lost throughout time lapse is picked up.

3. Metamorph tracked data 

4. Usiigaci tracker tracked data

	tracks.csv files are generated from the tracker. 
	One can specify one file, or in the folder mode, the script will looks for all nested folders each containing a tracks.csv file and automated data analysis will be carried out on all the tracks.csv files. 
	
of all, since Lineage mapper and Metamorph only provide cell centroids data, the parameters regarding cell area, perimeter and orientation cannot be analyzed (despite error the analysis should still be done, just lack of data during data analysis and visualization in the notebook)

## How to make your own training data and train them.
We manually annotate training data  (phase contrast image acquired on Nikon Ti-E microscope with 10X Ph-1 objective and 1.5X intermediate magnification on Hamamatsu Orca Flash V4.0 with 2x2 binning) using opensource software Fiji ImageJ.

1. manually outline cell into ROI

	Load the image into ImageJ and use the freehand tool to outline each cell into individual ROI and save into ROI manager. (a Wacom tablet or Apple ipad with apple pencil come in handy)

2. create instance masks.

	Use a plugin called LOCI from [University of Wisconsin](https://loci.wisc.edu), the ROI map function will index each individual ROI and output a 8bit indexed mask, save this mask as (labeled.png). 

	save a raw image file and annotated mask into individual folder as a set in each folder. 

3. run the preprocess_data.py to change the colored into gray scale 8 bit image. 
	Alternatively, if you already have the 8 bit gray scale image with each cell having its index. you're good to go by naming them as "instance_ids.png"


We used 50 sets of training data. you can find the training data we made in the train and val folder.
45 sets are used in training and 5 sets are for validation. We trained additional 200 epochs of headers and 300 epochs on all layers based on a trained network from Matterport with MS COCO dataset. 

The original ROIs used in the ImageJ also contains in the folder. You can use this to train on different neural network architect or on the [DeepCell](https://github.com/vanvalen/deepcell-tf)

We have found that Mask R-CNN network seems to be more resilient against environmental interferences in microscopy (out of focus, strong illumination, low light, etc) and the performance does not drop when segmenting cells with morphology that are significantly different from the cells in training set.

If the current trained network is suboptimal for you, be it poor accuracy or you have different size of images (which often need retraining of neural network), you can annotate your data by the same method and train further to see if it improves. 
