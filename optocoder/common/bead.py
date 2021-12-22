import numpy as np
import itertools

class Bead:
	"""Bead is the basic block of the spatial transcriptomics assay. Different ways of basecalling can create multiple attributes such as for
	the barcodes or the modified intensities.
		
		Args: 
			center (tuple): coordinates of the detected center of the bead
			radius (float): detected radius of the bead (this might not be perfectly accurate due to detection artifacts)

		Attributes:
			center (tuple): coordinates of the detected center of the bead
			radius (float): detected radius of the bead (this might not be perfectly accurate due to detection artifacts)
			barcode (dict): barcodes that are detected by different types of basecalling methods
			intensities (dict): bead intensities with raw and modified versions
		"""
	bead_id = itertools.count().__next__

	def __init__(self, center, radius, barcode_length):

		self.center =  center
		self.radius = radius 
		self.area = np.pi * self.radius**2

		self.barcode_length = barcode_length
		self.barcode = dict()
		self.scores = dict()
		self.intensities = {'raw': np.empty((barcode_length,4)), 
							'bc': np.empty((barcode_length,4)) ,
							'naive': np.empty((barcode_length,4)), 
							'naive_scaled': np.empty((barcode_length,4)), 
							'only_ct': np.empty((barcode_length,4)), 
							'only_ct_scaled': np.empty((barcode_length,4)), 
							'phasing': np.empty((barcode_length,4)), 
							'phasing_scaled': np.empty((barcode_length,4))
		}

		self.id = Bead.bead_id()

	def set_intensities(self, intensities, cycle, intensity_type):
		self.intensities[intensity_type][cycle] = intensities

	def __str__(self):
		return "Center of the bead :" + str(self.center) + "Radius of the bead :" + str(self.radius)
