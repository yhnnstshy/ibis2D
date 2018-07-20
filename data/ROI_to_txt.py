import os
from ij import IJ, WindowManager, ImageStack
from ij.gui import GenericDialog
from ij.plugin.frame import RoiManager
from ij.gui import Roi

PixelSize = 0.51190476
rootDir = "Your output root directory goes here"

def main():
	roi = RoiManager.getInstance()
	image = IJ.getImage()
	imageStack = image.getImageStack()
	RoiCount = roi.getCount()
	ImageCount = imageStack.getSize()
	
	tmp = image.getShortTitle().split('_')
	tmp.remove('DIC')
	tmp.append('xy')
	dirName = '_'.join(tmp)

	outDir = os.path.join(rootDir, dirName)

	if not os.path.exists(outDir):
		os.mkdir(outDir)

	if(RoiCount != ImageCount):
		print ("Warning: Not all Images have a corresponding ROI.")

	for i in range(0, RoiCount):
		filename = roi.getName(i).split('-')[0]
		filename = str(int(filename))
		roi.select(i)
		Poly=roi.getSelectedRoisAsArray()[0].getFloatPolygon()
		xPoints = Poly.xpoints
		yPoints = Poly.ypoints
		xCount = len(Poly.xpoints)
		yCount = len(Poly.ypoints)

		outfileName = outDir + '/' + dirName + '_' + filename + ".txt"

		outFile = open(outfileName, "w")

		if (xCount != yCount):
			print ("Error: something wrong with ROI file")
			return None
		for p in range(0, xCount):
			xPoint = PixelSize * xPoints[p]
			yPoint = PixelSize * yPoints[p]
			outFile.write("%f\t%f\n" %(xPoint,yPoint))

		outFile.close()

if __name__ in ['__builtin__','__main__']:
	main()
