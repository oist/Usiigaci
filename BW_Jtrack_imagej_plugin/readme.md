# BW JTrack

Very Simple Cells tracker, for Fiji.
Starting from a timelapse with masks (black and white),
It connect different masks to the same track based on maximum overlapping frame by frame.
It collect all the tracks as n_x_t_y where n is the number of the track and y is the frame as rois in roiManager
It starts to track selecting a white area and searching froward from that frame.


## Getting Started

Have a black and white already segmented stack

### Prerequisites

1. [Fiji](http://fiji.sc/) updated
2. [MorpholibJ Fiji Package](https://imagej.net/MorphoLibJ#Installation)


### Installing

1. Update your Fiji
2. Install MorphoLibJ via Update Site as Prerequisites
3. Relaunch Fiji
4. Copy BW_JtrackJ.py in the plugins folder of fiji
5. relaunch Fiji

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Emanuele Martini** - *Initial work* -


## License

Ask to the author
