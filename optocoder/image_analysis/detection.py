import cv2
import numpy as np
from skimage.exposure import adjust_gamma, equalize_hist
from skimage.util import img_as_float
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import convex_hull_image
from skimage import measure, io
import matplotlib.pyplot as plt
def get_convex_hull(im):
	"""Get the convex hull of the image"""

	overlay_image = img_as_float(im)

	#Blur the puck like crazy so that it would be just one big thing
	e = adjust_gamma(overlay_image, gamma=0.3)
	e = equalize_hist(e)
	e =  gaussian(e, sigma=10, mode='reflect')

	t = threshold_minimum(e)
	im = e >= t

	#get the biggest object which should be the puck
	labels_mask = measure.label(im)                       
	regions = measure.regionprops(labels_mask)
	regions.sort(key=lambda x: x.area, reverse=True)
	if len(regions) > 1:
		for rg in regions[1:]:
			labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
	labels_mask[labels_mask!=0] = 1
	mask = labels_mask

	return mask

def detect_beads(overlay_image):
	"""Detect the beads using Hough Circle Transform
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html

	Args:
		overlay_image (ndarray): Image to detect the beads
	"""

	#Some blurring for noise cleanup
	img = cv2.medianBlur(overlay_image,3)
	
	#Adaptive histogram equalization for better contrast
	clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(64,64))
	img = clahe.apply(img)

	#Get the convex hull of the image to filter out the detections outside of the pucl
	hull = get_convex_hull(overlay_image)

	#Detect the circles
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,8,
									param1=50,param2=10,minRadius=4,maxRadius=10)
	circles = np.uint16(np.around(circles))

	#Filter out the outside detections
	puck_beads = []
	for i in circles[0,:]:
		if hull[i[1]-1,i[0]-1] != 0:
			puck_beads.append(i)

	return puck_beads

def get_intensity(bead, image):
	"""Get the intensity value for a bead in a channel"""

	try:
		x = int(bead.center[0])
		y = int(bead.center[1])

		#radius = int(bead.radius)
		radius = 3

		#Get the average bead intensity
		crop_img = image[(y-radius):(y+radius), (x-radius):(x+radius)]

		if crop_img.size == 0:
			mean = np.NaN
		else:
			mean = np.mean(crop_img)
		if np.isnan(mean):
			return -1
		else:
			return mean
	except:
		return -1