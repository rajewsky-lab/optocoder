import cv2
import numpy as np

def register_image(reference, unaligned_image, channels):
	"""Image registration using Enhanced Cross Correlation (ECC) maximization algorithm 
	(http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf)

	Args: 
		reference (ndarray): This is the target image that we will register to
		image (ndarray): This is the image that will be transformed
		channels (list): Channel images

	Returns:
		transformed (ndarray): Warped (registered) image
		transformed_channels (list): Warped (registered) channel images
	"""

	size = reference.shape

	#Euclidian motion model includes translation and rotation, cv2.MOTION_AFFINE can be used for shear and scale
	warp_mode = cv2.MOTION_EUCLIDEAN
	warp_matrix = np.eye(2, 3, dtype=np.float32)

	#Optimization parameters
	number_of_iterations = 2000;
	termination_eps = 1e-6;
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
	
	#We do image pyramiding for better performance. This means downsampling the image and improving warp matrix 
	#step by step
	nol = 8 #Number of levels
	warp = warp_matrix
	warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)**(1-nol)

	#Iterate for every pyramid level
	for level in range(nol):
	    #Scale the image
	    scale = 1/2**(nol-1-level)
	    resized_img1 = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
	    resized_img2 = cv2.resize(unaligned_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

	    #Find the warp function
	    cc, warp = cv2.findTransformECC(resized_img1, resized_img2, warp, warp_mode, criteria, None, 1)

	    #Go to the upper level
	    if level != nol-1:
	    	warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)

	#Warp the source image
	transformed = cv2.warpAffine(unaligned_image, warp, (size[1],size[0]), flags=cv2.WARP_INVERSE_MAP)
	transformed_channels = []

	#Warp channel images
	for channel in channels: 
		tf_ch = cv2.warpAffine(channel, warp, (size[1],size[0]), flags=cv2.WARP_INVERSE_MAP)
		transformed_channels.append(tf_ch)

	return transformed, transformed_channels, warp

def get_warped_image(image, warp):
	size = image.shape
	return cv2.warpAffine(image, warp, (size[1],size[0]), flags=cv2.WARP_INVERSE_MAP)
