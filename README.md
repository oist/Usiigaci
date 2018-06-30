# Usiigaci: Label-free instance-aware cell tracking under phase contrast microscopy using Mask-RCNN
Hsieh-Fu Tsai<sup>1,2</sup>, Tyler Sloan<sup>3</sup>, Joanna Gajda<sup>4</sup>, and Amy Q. Shen<sup>1</sup>

<sup>1</sup>Micro/Bio/Nanofluidics Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa Japan
<sup>2</sup>Research Fellow of Japan Society of Promotion for Science
<sup>3</sup>Quorumetrix Solutions, Canada
<sup>4</sup>affil

![T98G microscopy](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gelectrotaxis-1.gif)
![T98G results from Usiigaci](https://github.com/oist/Usiigaci/blob/master/Demo/T98Gmask-1.gif)

Usiigaci means "tracing", "描き写し" in Ryukyuan language. 

Zernike's phase contrast microscopy is a brightfield microscopy technique developed by Frits Zernike and by inveting the phase contrast technique, he won the 1953 Nobel Prize for physics. Phase contrast microscopy is favored by biologists because it translates the phase difference caused by cell components into amplitude thus making these transparent structures more visible. Also, in comparison to differential interference microscopy, phase contrast microscopy works without problems with different substrates especially on plastics that contains high birefringence. 

However, phase contrast microscopy images are notoriously difficult to segment by conventional computer vision methods. Accurate single cell tracking is the hallmark of cell migration microscopy imaging. Accurate whole cell outline segmentation and resolution of cells that contact each other is essential for accurate cell migration analysis. 

Here we report Usiigaci, a semi-automated pipeline to segment, track, and visualize cell migration in phase contrast microscopy.

High accuracy label-free instance-aware segmentation is achieved by adapting the mask regional convolutional neural network (Mask R-CNN), winner of Marr prize at ICCV 2017 by He *et al.*. We built our segmentation part on the Mask R-CNN implementation by [Matterport](https://github.com/matterport/Mask_RCNN). High accuracy whole cell segmentation allow us to analyze both cell migration and cell morphology which is previously difficult without fluorescence imaging. 


Cell tracking and data verification can be done in ImageJ or other tracking software such as [Lineage Mapper](https://github.com/usnistgov/Lineage-Mapper).

A Jupyter Notebook and a python script is developed for automated processing and visualization of the tracked results. Step-centric and cell-centric parameters are automatically computed and saved into additional spreadsheets where users can access and reuse in statistical software or R. Automated visualization of cell migration is also generated for cell trajectory graph, box plots, etc. 

We worked on Usiigaci for our work on cell electrotaxis study, and hopefully can devote to current international trend to standardize cell migration experiments. 

Usiigaci is released under MIT License. 

We hope Usiigaci is interesting to you and if it is useful, please cite the following paper.


## Dependencies
### Mask-RCNN
* Kubuntu 16.04 Linux
* NVIDIA graphics card with compute capability > 5.0
* CUDA 9.1
* TensorFlow 1.4
* Keras 2.1.2

### Single cell migration data analysis notebook
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


## How to make your own training data and train them.
We manually annotate training data  (phase contrast image acquired on Nikon Ti-E microscope with 10X Ph-1 objective and 1.5X intermediate magnification on Hamamatsu Orca Flash V4.0 with 2x2 binning) using opensource softare Fiji ImageJ.
1. manually outline cell into ROI
Load the image into ImageJ and use the freehand tool to outline each cell into individual ROI and save into ROI manager. (a Wacom tablet or Apple ipad with apple pencil come in handy)
2. create instance masks.
Use a pluging call LOCI, the ROI map function will index each individual ROI and output a 8bit indexed mask. 
save a raw image file and annotated mask into individual folder as a set. 

We used 50 sets of training data.
45 sets are used in training and 5 sets are for validation. We trained additional 200 epochs of headers and 300 epochs on all layers based on a trained network from Matterport with MS COCO dataset. 

We have found that mask-rcnn network seems to be more resilient against environmental interferences in microscopy (out of focus, strong illumination, not enough of illumination) and the performance does not drop when segmenting cells with morphology that significantly different from the cells in training set.

If the current trained network is suboptimal for you, be it poor accuracy or you have different size of images (which often need retraining of neural network), you can annotate your data by the same methods and train further see if it improves. 