# Usiigaci: Label-free instance-aware cell tracking under phase contrast microscopy using Mask R-CNN
Hsieh-Fu Tsai<sup>1,2</sup>, Joanna Gajda<sup>3</sup>, Tyler Sloan<sup>4</sup>, Andrei Rares<sup>5</sup>, and Amy Q. Shen<sup>1</sup>

<sup>1</sup>Micro/Bio/Nanofluidics Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa Japan
<sup>2</sup>Research Fellow of Japan Society of Promotion for Science
<sup>3</sup>
<sup>4</sup>Quorumetrix Solutions, Canada
<sup>5</sup>affil

![T98G microscopy](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gelectrotaxis-1.gif)
![T98G results from Usiigaci](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gmask-3.gif)

[Usiigaci](http://ryukyu-lang.lib.u-ryukyu.ac.jp/srnh/details.php?ID=SN03227) in Ryukyuan language means "tracing", "透き写し" in Japanese，*i.e.* drawing the outline of objects based on a template. The process is essentially what we do: following the morphology and position of cells under microscope, analyze how cell respond upon environmental perturbation in the microenvironment. However, this process is bloody tedious if done by human, and now we developed a pipeline using the famous Mask R-CNN to do this for us. Letting us not only track objects by their position but also track how their morphology changes through time. 

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

Usiigaci is released under MIT License. 

We hope Usiigaci is interesting to you and if it is useful for your research, please cite the following paper.
```
Hsieh-Fu Tsai, Joanna Gajda, Tyler Sloan, Andrei Rares, and Amy Q. Shen, softwareX, inprep
```

## Dependencies
### Hardware
A computer with CUDA-ready GPU should be able to run Usiigaci. 
We have built all the testing and development on an Alienware 15 laptop with GTX1070 8GB GPU.


### Mask R-CNN
* Kubuntu 16.04 Linux
* NVIDIA graphics card with compute capability > 5.0
* CUDA 9.1
* TensorFlow 1.4
* Keras 2.1.2

### Python tracking GUI
* trackpy
* scikit-image
* Numpy
* Pandas
* Matplotlib
* PIMS

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
### Segmentation using Mask R-CNN

1. Download the trained weights
	
	Due to the file size limit of Github, please download the three weights we used for our phase contrast microscope in the [Dropbox folder](https://www.dropbox.com/sh/3eldgvytfchm9gr/AAB6vzPaEf8buk81IRVNClUEa?dl=0).

	Note: These are trained for phase contrast images on Nikon Ti-E with 10X phase contrast objective, 1.5X intermediate magnification with a Hamamatsu Orca Flash V4.0 sCMOS camera at 1024x1022 size. We have found Mask R-CNN to be more resilient toward environmental changes, but if the results from pretrained weights are suboptimal, you can see the last section to train the network with your own data.




The inference script "/Mask-RCNN/Inference.py" is the script you need to run on images for generating corresponding masks. 
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

A python tracking software is developed using the [trackpy](hhttps://soft-matter.github.io/trackpy/v0.3.2/) by Dr. Andrei Rares. 
1. The segmented masks from Mask R-CNN are loaded and tracked using trackpy.
2. The tracking results that are suboptimal from segmentation error were repaired. 
3. A GUI is used to allow user to double check the results and deleted bad tracks if they exists.
4. The XY coordinate, area, perimeter, and angle (in radians) is extracted using scikit-image regionprops methods.
5. the results are saved into "tracks.csv" files and tracked images as well as movies are saved also. 

#### Tracking using other tracking software

Alternatively, you can load the indexed 8 bit masks files into Lineage Mapper, or Metamorph which the tracking can be eaily done.

### Data analysis and visualization
Currently we have coded data loading interfaces for 5 types of analyzed data. but you can look into the codes to adapt the output data in your preference also.
1. ImageJ tracked multi-measure output 

	each cell track of all time points followed by another

2. trackpy output data by our python tracker
	
	each cell track of all time points followed by another. Users can use the GUI to delete bad tracks for data verification.

3. Lineage Mapper tracked results

	only cells that did not divide, fusion or lost throughout time lapse is picked up.

4. Metamorph tracked data 

5. Usiigaci tracked data

	tracks.csv files are generated from the tracker. 
	One can specify one file, or a folder containing many tracks.csv files. Automated data analysis is carried out on all the tracks.csv files. 
	
of all, since Lineage mapper and Metamorph only provide cell centroids data, the parameters regarding cell area, perimeter and orientation cannot be analyzed (there shouldn't be any error, just lack of data during data analysis and visualization in the notebook)

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