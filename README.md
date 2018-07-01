# Usiigaci: Label-free instance-aware cell tracking under phase contrast microscopy using Mask-RCNN
Hsieh-Fu Tsai<sup>1,2</sup>, Tyler Sloan<sup>3</sup>, Joanna Gajda<sup>4</sup>, and Amy Q. Shen<sup>1</sup>

<sup>1</sup>Micro/Bio/Nanofluidics Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa Japan
<sup>2</sup>Research Fellow of Japan Society of Promotion for Science
<sup>3</sup>Quorumetrix Solutions, Canada
<sup>4</sup>affil

![T98G microscopy](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gelectrotaxis-1.gif)
![T98G results from Usiigaci](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gmask-1.gif)

[Usiigaci](http://ryukyu-lang.lib.u-ryukyu.ac.jp/srnh/details.php?ID=SN03227) in Ryukyuan language means "tracing", "透き写し"，*i.e.* drawing the outline of objects on a template. The process is essentially what we do: following the morphology and position of cells under microscope, analyze what they do upon changes in microenvironment.

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
### hardware
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

### data verification/tracking

### data analysis and visualization
Currently we have finished data loading interface for three type of analyzed data
1. ImageJ tracked multi-measure output 

	each cell track of all time points followed by another

2. Lineage Mapper tracked results

	only cells that did not divide, fusion or lost throughout time lapse is picked up.

3. Metamorph tracked data 
	
of all, since Lineage mapper and Metamorph only provide cell centroids data, the parameters regarding cell area, perimeter and orientation cannot be analyzed (there shouldn't be any error, just lack of data)

## How to make your own training data and train them.
We manually annotate training data  (phase contrast image acquired on Nikon Ti-E microscope with 10X Ph-1 objective and 1.5X intermediate magnification on Hamamatsu Orca Flash V4.0 with 2x2 binning) using opensource software Fiji ImageJ.

1. manually outline cell into ROI

	Load the image into ImageJ and use the freehand tool to outline each cell into individual ROI and save into ROI manager. (a Wacom tablet or Apple ipad with apple pencil come in handy)

2. create instance masks.

	Use a plugin called LOCI from [University of Wisconsin](https://loci.wisc.edu), the ROI map function will index each individual ROI and output a 8bit indexed mask, save this mask as (labeled.png). 

save a raw image file and annotated mask into individual folder as a set in each folder. 

We used 50 sets of training data.
45 sets are used in training and 5 sets are for validation. We trained additional 200 epochs of headers and 300 epochs on all layers based on a trained network from Matterport with MS COCO dataset. 

We have found that Mask RCNN network seems to be more resilient against environmental interferences in microscopy (out of focus, strong illumination, low light, etc) and the performance does not drop when segmenting cells with morphology that are significantly different from the cells in training set.

If the current trained network is suboptimal for you, be it poor accuracy or you have different size of images (which often need retraining of neural network), you can annotate your data by the same method and train further see if it improves. 