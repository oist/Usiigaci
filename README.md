# Usiigaci
Hsieh-Fu Tsai<sup>1,2</sup>, Tyler Sloan<sup>3</sup>, Joanna Gajda<sup>4</sup>, and Amy Q. Shen<sup>1</sup>

<sup>1</sup>Micro/Bio/Nanofluidics Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa Japan
<sup>2</sup>Research Fellow of Japan Society of Promotion for Science
<sup>3</sup>Quorumetrix Solutions, Canada
<sup>4</sup>affil


Label-free instance-aware cell tracking under phase contrast microscopy using Mask-RCNN

Usiigaci means "tracing", "描き写し" in Ryukyuan language. 

Zernike's phase contrast microscopy is a brightfield microscopy technique developed by Frits Zernike and by inveting the phase contrast technique, he won the 1953 Nobel Prize for physics. Phase contrast microscopy is favored by biologists because it translates the phase difference caused by cell components into amplitude thus making these transparent structures more visible. Also, in comparison to differential interference microscopy, phase contrast microscopy works without problems with different substrates especially on plastics that contains high birefringence. 

However, phase contrast microscopy images are notoriously difficult to segment by conventional computer vision methods. Accurate single cell tracking is the hallmark of cell migration microscopy imaging. Accurate whole cell outline segmentation and resolution of cells that contact each other is essential for accurate cell migration analysis. 

Here we report Usiigaci, a semi-automated pipeline to segment, track, and visualize cell migration in phase contrast microscopy.

High accuracy label-free instance-aware segmentation is achieved by adapting the mask regional convolutional neural network (Mask R-CNN), winner of Marr prize at ICCV 2017 by He *et al.*. We used the Mask R-CNN implementation by Matterport to streamline experiment. High accuracy whole cell segmentation allow us to analyze both cell migration and cell morphology which is previously difficult without fluorescence imaging. 


Cell tracking and data verification can be done in ImageJ or other tracking software such as Lineage Mapper.

A Jupyter Notebook and a python script is developed for automated processing and visualization of the tracked results. Step-centric and cell-centric parameters are automatically computed and saved into additional spreadsheets where users can access and reuse in statistical software or R. Automated visualization of cell migration is also generated for cell trajectory graph, box plots, etc. 

We worked on Usiigaci for our work on cell electrotaxis study, and hopefully can devote to current international trend to standardize cell migration experiments. 

Usiigaci is released under MIT License. 

We hope Usiigaci is interesting to you and if it is useful, please cite the following paper.


## Dependencies



## How to use Usiigaci 


## How to make your own training data