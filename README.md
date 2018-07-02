# Usiigaci: Label-free instance-aware cell tracking under phase contrast microscopy using Mask-RCNN
Hsieh-Fu Tsai<sup>1,2</sup>, Tyler Sloan<sup>3</sup>, Joanna Gajda<sup>4</sup>, and Amy Q. Shen<sup>1</sup>

<sup>1</sup>Micro/Bio/Nanofluidics Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa Japan
<sup>2</sup>Research Fellow of Japan Society of Promotion for Science
<sup>3</sup>Quorumetrix Solutions, Canada
<sup>4</sup>affil

![T98G microscopy](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gelectrotaxis-1.gif)
![T98G results from Usiigaci](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gmask-1.gif)

[Usiigaci](http://ryukyu-lang.lib.u-ryukyu.ac.jp/srnh/details.php?ID=SN03227) in Ryukyuan language means "tracing", "透き写し"，*i.e.* drawing the outline of objects on a template. The process is essentially what we do: following the morphology and position of cells under microscope, analyze what they do upon changes in microenvironment. It's just bloody tedious to do this by human, and now we developed a pipeline using the famous Mask-RCNN to do this for us. Letting us not only track objects by their position but also how their morphology changes through time. 

Zernike's phase contrast microscopy is a brightfield microscopy technique developed by Frits Zernike and by inventing the phase contrast technique, he won the 1953 Nobel Prize for physics. Phase contrast microscopy is favored by biologists because it translates the phase difference caused by cell components into amplitude thus making these transparent structures more visible. Also, in comparison to differential interference microscopy, phase contrast microscopy works without problems with different substrates especially on plastics that contains high birefringence. 

However, phase contrast microscopy images are notoriously difficult to segment by conventional computer vision methods. Accurate single cell tracking is the hallmark of cell migration microscopy imaging. Accurate whole cell outline segmentation and resolution of cells that contact each other is essential for accurate cell migration analysis. 

Here we report Usiigaci, a semi-automated pipeline to segment, track, and visualize cell migration in phase contrast microscopy.

High accuracy label-free instance-aware segmentation is achieved by adapting the mask regional convolutional neural network (Mask R-CNN), winner of Marr prize at ICCV 2017 by He *et al.*. We built our segmentation part on the Mask R-CNN implementation by [Matterport](https://github.com/matterport/Mask_RCNN). High accuracy whole cell segmentation allow us to analyze both cell migration and cell morphology which is previously difficult without fluorescence imaging. 


Cell tracking and data verification can be done in ImageJ or other tracking software such as [Lineage Mapper](https://github.com/usnistgov/Lineage-Mapper).

A Jupyter Notebook and a python script is developed for automated processing and visualization of the tracked results. Step-centric and cell-centric parameters are automatically computed and saved into additional spreadsheets where users can access and reuse in statistical software or R. Automated visualization of cell migration is also generated for cell trajectory graph, box plots, etc. 
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

We worked on Usiigaci for our work on cell electrotaxis study, and hopefully can devote to current international trend to standardize cell migration experiments. 

Usiigaci is released under MIT License. 

We hope Usiigaci is interesting to you and if it is useful, please cite the following paper.
```
Hsieh-Fu Tsai, Tyler Sloan, Joanna Gajda, and Amy Q. Shen, softwareX, inprep
```

## Dependencies
### Hardware
A computer with CUDA-ready GPU should be able to do.
We have built all the testing and development on an Alienware 15 with GTX1070 8GB laptop.

### Mask-RCNN
* Kubuntu 16.04 Linux
* NVIDIA graphics card with compute capability > 5.0
* CUDA 9.1
* TensorFlow 1.4
* Keras 2.1.2

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
### Segmentation using Mask RCNN
The inference script "/Mask-RCNN/Inference.py" is the script you need to run prediction on images.
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
1. Load the images in ImageJ as a stack, you can verify the data and do manual tracking.
2. for ease of tracking, you can use an ImageJ plugin developed by Emanuele Martini. 
	1. threshold the instance-aware stack to binary masks.
	2. run the plugin, you can find one target cells on the first slice with magic wand tool and click ok, based on overlapping, the plugin will find the target ROIs in the rest of the slices and add them into ROI manager.
	3. in ROI manager, you can edit each ROI and click "multimeasure" to output measured results for further analysis.

3. Alternatively, you can load the indexed 8 bit masks files into Lineage Mapper, or Metamorph which the tracking can be eaily done.

### Data analysis and visualization
Currently we have finished data loading interface for three type of analyzed data
1. ImageJ tracked multi-measure output 

	each cell track of all time points followed by another

2. Lineage Mapper tracked results

	only cells that did not divide, fusion or lost throughout time lapse is picked up.

3. Metamorph tracked data 
	
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

We have found that Mask RCNN network seems to be more resilient against environmental interferences in microscopy (out of focus, strong illumination, low light, etc) and the performance does not drop when segmenting cells with morphology that are significantly different from the cells in training set.

If the current trained network is suboptimal for you, be it poor accuracy or you have different size of images (which often need retraining of neural network), you can annotate your data by the same method and train further see if it improves. 