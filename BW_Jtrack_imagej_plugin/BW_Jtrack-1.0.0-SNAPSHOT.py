'''
Created on Apr 16, 2018

Very Simple Cells tracker, for Fiji.
Starting from a timelapse with masks (black and white),
It connect different masks to the same track based on maximum overlapping frame by frame.
It collect all the tracks as n_x_t_y where n is the number of the track and y is the frame as rois in roiManager
It starts to track selecting a white area and searching froward from that frame.

@author: emartini
@email: emanuele.martini00@gmail.com
'''


from ij import IJ, WindowManager, ImagePlus
from ij.gui import WaitForUserDialog,  GenericDialog
from ij.measure import Measurements
from ij.plugin import Duplicator
from ij.plugin.filter import ParticleAnalyzer as PA
from ij.plugin.frame import RoiManager
from java.awt import Color
from ij.gui import ShapeRoi
from java.util import Random


def getTrack(imp,roi_selected,n_track,searching_radius,iter_number,track_color):
    DUPL = Duplicator()
    nSlices = imp.getNSlices()

    rm = RoiManager().getInstance()


    z = imp.getZ()

    if z<nSlices:
        IJ.run(imp, "Select None", "");
        imp_track_z_1 = DUPL.run(imp,1,1,z+1,z+1,1,1)
        roi_selected.setName("n_"+str(n_track)+"_t_"+str(z))
        if iter_number==1:
            roi_selected.setStrokeColor(track_color)
            rm.addRoi(roi_selected)

        current_rois = rm.getCount()
        imp_track_z_1.show()
        imp_track_z_1.setRoi(roi_selected)
        IJ.run("Clear Outside")
        IJ.run(imp_track_z_1, "Select None", "");
        roi_z_1=None
        if imp_track_z_1.getStatistics().mean>0:
            IJ.run(imp_track_z_1, "Keep Largest Region", "");
            IJ.run(imp_track_z_1, "Invert", "");
            largest_mask = WindowManager.getImage(imp_track_z_1.getTitle()+"-largest")
            IJ.run(largest_mask, "Invert", "");
            IJ.run(largest_mask, "Create Selection", "");
            roi_and = largest_mask.getRoi()
            centroid = roi_and.getContourCentroid()
            pa = PA(PA.ADD_TO_MANAGER, Measurements.AREA, None, 0,float('inf'))
            pa.setHideOutputImage(True)
            pa.setRoiManager(rm)
            imp.setZ(z+1)
            IJ.setAutoThreshold(imp, "Default dark");
            pa.analyze(imp)
            largest_mask.hide()
            imp_track_z_1.hide()
            for roi in rm.getInstance().getRoisAsArray():
                check_roi = ShapeRoi(roi_and).and(ShapeRoi(roi))
                if check_roi.boundingRect.width>0:
                    roi_z_1 = roi
                    #roi_z_1.setPosition(z+1)
                    roi_z_1.setName("n_"+str(n_track)+"_t_"+str(z+1))



            if roi_z_1 == None:
                roi_z_1 = roi_selected
                roi_z_1.setPosition(z+1)
                #roi_z_1.setPosition(z+1)
                roi_z_1.setName("n_"+str(n_track)+"_t_"+str(z+1))

            idx_to_delete = range(current_rois,rm.getCount())
            rm.setSelectedIndexes(idx_to_delete)
            rm.runCommand(imp,"Delete");
            imp.setZ(z+1)
            roi_z_1.setStrokeColor(track_color)
            imp.setRoi(roi_z_1)
            rm.addRoi(imp.getRoi())





        else:
            roi_z_1 = roi_selected
            #roi_z_1.setPosition(z+1)
            roi_z_1.setName("n_"+str(n_track)+"_t_"+str(z+1))
            roi_z_1.setPosition(z+1)
            imp.setZ(z+1)
            roi_z_1.setStrokeColor(track_color)
            imp.setRoi(roi_z_1)
            rm.addRoi(imp.getRoi())
        imp_track_z_1.hide()
        c = 1
        imp.setZ(z+1)
        getTrack(imp, roi_z_1, n_track, searching_radius,iter_number+1,track_color)


def run_script():
    searching_radius = 10 #to be implemented
    rm = RoiManager().getInstance()
    if rm == None:
    	rm = RoiManager()
    imp = IJ.getImage()
    IJ.run(imp, "Select None", "");
    IJ.run(imp, "8-bit", "");
    IJ.setAutoThreshold(imp, "Default dark");
    # area opening to remove
    IJ.run(imp, "Analyze Particles...", "size=100-100000 show=Masks exclude in_situ stack");


    doAnalysis = True

    if rm.getCount()==0:
    	n_track = 1
    else:
		last_roi = rm.getRoi(rm.getCount()-1)
		last_name =last_roi.getName()
		track_number = last_name[2]
		n_track = int(track_number)+1


    while(doAnalysis):
        imp.setSlice(1)

        IJ.setTool("wand")
        WaitForUserDialog("SELECT A ROI", "select with wand to track").show()
        Rand = Random()
        hue = Rand.nextFloat();
        saturation =float(Rand.nextInt(2000) + 1000) / float(10000)
        luminance = 0.9
        track_color = Color.getHSBColor(hue, saturation, luminance)

        getTrack(imp,imp.getRoi(),n_track,searching_radius,1,track_color)
        n_track = n_track+1;

        gd = GenericDialog("CONTINUE")
        gd.addMessage("Track another roi?")
        gd.showDialog()
        doAnalysis = gd.wasOKed()

    IJ.log("finish")
    IJ.run(imp, "Select None", "");
    rm.runCommand(imp,"Show None");

    IJ.resetThreshold(imp);
    IJ.setTool("rectangle");

if __name__ in ['__builtin__', '__main__']:
    run_script();
