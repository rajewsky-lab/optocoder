import optocoder.image_analysis.detection as detection
import cv2
import numpy as np
import multiprocessing
import pickle
import os
import yaml
import pandas as pd

#Opseq module imports
from optocoder.common.bead import Bead
from optocoder.evaluation.evaluate_basecalls import plot_selected, plot_barcode_confidences, plot_barcode_compression, plot_barcode_entropy, evaluate_cycles, evaluate_fractions, output_prediction_data
from optocoder.evaluation.evaluate_images import save_intensity_data, save_detected_beads, get_cross_section_profile, evaluate_channel_intensities, save_registered_images
from optocoder.evaluation.report import create_report
from optocoder.evaluation.evaluate_registration import calculate_ssim
from optocoder.evaluation.evaluate_spatial import plot_barcodes_in_space, plot_entropy_in_space, plot_compression_in_space, plot_chastity_in_space

import logging

class Experiment():
	"""Implements an experiment to manage bead detection and basecalling. This module
		will be extended to deal with multiple slices etc. 
		
		Args: 
			puck_id (str): Id for the experiment that can be used for reporting purposes
			num_cycles (int): number of cycles
			image_manager (ImageManager): ImagerManager object that deals with the microscopy images
		

		Attributes:
			puck_id (str): Id for the experiment that can be used for reporting purposes
			num_cycles (int): number of cycles
			image_manager (ImageManager): ImagerManager object that deals with the microscopy images
			beads (list): List of beads in the experiments
		"""

	def __init__(self, puck_id, num_cycles, image_manager):

		self.puck_id = puck_id
		self.num_cycles = num_cycles
		self.image_manager = image_manager
		self.methods = ['naive', 'only_ct', 'phasing']
		self.beads = []
		self.unique_barcodes = dict()

	def detect_beads(self):
		"""Detect beads from the last cycle using Hough Transform"""

		# Get the overlay image for the last cycle
		last_cycle_image = self.image_manager.get_overlay(cycle=self.num_cycles-1)
		
		# Detect the beads
		circles = detection.detect_beads(last_cycle_image)

		# Iterate every detected circle
		for i in circles:
			#Create a bead with x,y,radius
			bead = Bead((i[0],i[1]),i[2], self.num_cycles)
			self.beads.append(bead)

	def calculate_bead_intensities(self):
		"""Calculate the intensity for a bead in every cycle"""

		for cycle in range(self.num_cycles):
			logging.info(f'Calculating bead intensities for cycle {cycle}...')

			logging.info(f'Registering cycle {cycle}.')
			# register the cycle 
			overlay, channels = self.image_manager.register_image_to_cycle(cycle_id=cycle, reference_cycle_id=-1)

			logging.info(f'Correcting background for cycle {cycle}.')

			# apply background correction
			corrected_channels = self.image_manager.correct_background(channels)

			for bead in self.beads:
				cycle_ints = []
				for channel in corrected_channels:
					channel_intensity = detection.get_intensity(bead, channel)
					cycle_ints.append(channel_intensity)
				bead.set_intensities(cycle_ints, cycle, 'bc')
		

			logging.info(f'Bead intensities are detected for {cycle}.')

	def base_call(self, basecaller):
		"""Call the bases"""
		basecaller.base_call(self.beads)

	def save_experiment(self, output_orig):
		with open(os.path.join(output_orig, 'experiment.pkl'), 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def get_bead_frame(self, method):
		beads = self.beads
		bead_ids = [bead.id for bead in beads]
		locxs = [bead.center[0] for bead in beads]
		locys = [bead.center[1] for bead in beads]
		barcodes = [bead.barcode[method] for bead in beads]

		d = {"bead_id": bead_ids, "x_pos": locxs, "y_pos": locys, "barcodes": barcodes}
		df = pd.DataFrame(d)

		confidences = [bead.scores[method] for bead in beads]
		confidences = [np.hstack(c) for c in confidences]

		df_scores = pd.DataFrame(confidences, columns=['score_cycle_%i_nuc_%s' % (i,j) for i in range(1,self.num_cycles+1) for j in ['G', 'T', 'A', 'C']])
			
		df = pd.concat([df, df_scores], axis=1)
		return df

	def report(self, output_orig):
		
		# calculate  number of beads and unique barcodes
		self.num_beads = len(self.beads)
		self.unique_barcodes['phasing'] = len(set([bead.barcode['phasing'] for bead in self.beads]))
		self.unique_barcodes['naive'] = len(set([bead.barcode['naive'] for bead in self.beads]))
		self.unique_barcodes['only_ct'] = len(set([bead.barcode['only_ct'] for bead in self.beads]))

		# set the output folders
		output_plot_path = os.path.join(output_orig, 'plots')
		output_intensities_path = os.path.join(output_orig, "intensity_files")
		if not os.path.exists(output_plot_path):
			os.mkdir(output_plot_path)
		if not os.path.exists(output_intensities_path):
			os.mkdir(output_intensities_path)
		
		save_registered_images(self.image_manager, self.beads, output_orig)

		for method in self.methods:
			evaluate_fractions(self.beads, self.num_cycles, output_plot_path, method=method)
			output_prediction_data(self.beads, self.num_cycles, output_orig, method=method)
			plot_barcode_entropy(self.beads, output_plot_path, method=method)
			plot_barcode_compression(self.beads, output_plot_path, method=method)
			plot_barcode_confidences(self.beads, self.num_cycles, output_plot_path, method=method)
			plot_entropy_in_space(self.get_bead_frame(method), self.num_cycles, output_plot_path)	
			plot_compression_in_space(self.get_bead_frame(method), self.num_cycles, output_plot_path)	
			plot_chastity_in_space(self.get_bead_frame(method), self.num_cycles, output_plot_path)
			plot_barcodes_in_space(self.get_bead_frame(method), self.num_cycles, output_plot_path)
			
		#Save intensity data for raw, background corrected and normalized data
		save_intensity_data('raw', self.beads, output_intensities_path)
		save_intensity_data('bc', self.beads, output_intensities_path)
		save_intensity_data('only_ct', self.beads, output_intensities_path)
		save_intensity_data('phasing', self.beads, output_intensities_path)

		#Save image analysis
		calculate_ssim(self.image_manager, self.num_cycles, output_plot_path)
		#get_cross_section_profile(self.image_manager, self.num_cycles, output_plot_path)
		#evaluate_channel_intensities(self.image_manager, self.num_cycles, output_plot_path)
		
		save_detected_beads(self.image_manager, self.beads, output_plot_path)

		#Save barcode and basecalling analysis

		#plot_selected(self.beads, output_plot_path)
		evaluate_cycles(self.beads, self.num_cycles, output_plot_path, ['phasing', 'naive','only_ct'])

		#save predictions and the pdf report
		#output_prediction_data(self.beads, self.num_cycles, output_orig)
		with open(os.path.join(output_orig, 'report_summary.yaml'), 'w') as outfile:
			yaml.dump({
				'num_cycles': self.num_cycles,
				'puck_id': self.puck_id,
				'num_beads': self.num_beads,
				'unique_barcodes': self.unique_barcodes
			}, outfile, default_flow_style = False)

		#create_report(output_orig=output_orig, output_plot_path=output_plot_path, report_name='report.pdf')