import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.draw import line
from skimage.measure import profile_line
from optocoder.image_analysis.registration import get_warped_image

plt.rcParams.update({'font.size': 24, 'lines.linewidth':4})
import itertools

def save_intensity_data(intensity_type, beads, output_path):

	bead_ids = [bead.id for bead in beads]
	locxs = [bead.center[0] for bead in beads]
	locys = [bead.center[1] for bead in beads]

	#save initial intensities
	intensities = []

	intensities = [np.asarray(bead.intensities[intensity_type]).flatten() for bead in beads]
	intensities = np.vstack(intensities)

	d = {"bead_id": bead_ids, "x_pos": locxs, "y_pos": locys}
	df = pd.DataFrame(d)

	df_intensities = pd.DataFrame(intensities, columns=['cycle_%i_ch_%i' % (i,j) for i in range(1,beads[0].barcode_length+1) for j in range(1,5)])

	df = pd.concat([df, df_intensities], axis=1)

	df.to_csv(os.path.join(output_path,'%s_intensities.csv' % intensity_type), index=False, sep='\t', float_format='%.4f')

def evaluate_channel_intensities(image_manager, num_cycles, output_path):
	intensities = []
	for cycle in image_manager.cycles:
		cycle_channels = cycle['raw_channels']
		mean_cycle_intensities = [np.mean(ch) for ch in cycle_channels]
		intensities.append(mean_cycle_intensities)
	colors = ["#9467bd", "#ff7f0e", "#7f7f7f", '#8c564b', '#2ca02c', '#d62728']
	intensities = np.matrix(intensities)
	f = plt.figure(figsize=(16,10))
	for y, color in zip(intensities.T, colors):
		plt.plot(np.arange(1,num_cycles+1), y.T, '-o', color=color)
	plt.ylabel("Intensity")
	plt.xlabel("Cycles")
	plt.xticks(np.arange(1,num_cycles+1))
	plt.legend(["Ch %i" % i for i in np.arange(1, 5)])

	plt.savefig(os.path.join(output_path, 'average_intensities.png'),bbox_inches='tight',dpi=150)
	plt.close()

def save_image(image, cycle_id, output_path, folder):
	output_path = os.path.join(output_path, "intermediate_files", "images", folder)
	os.makedirs(output_path, exist_ok=True)
	cimg = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
	cv2.imwrite(os.path.join(output_path, 'cycle_%i.png' % cycle_id), cimg)

def save_registered_images(image_manager, beads, output_path):
	output_path = os.path.join(output_path, "intermediate_files", "images", 'registered_images_with_beads')
	os.makedirs(output_path, exist_ok=True)

	for i, warp in enumerate(image_manager.warp_params):
		warped_image = get_warped_image(image_manager._read_image(i)[1], warp)
		cimg = cv2.cvtColor(warped_image,cv2.COLOR_GRAY2BGR)

		for bead in beads:
			cv2.circle(cimg,(bead.center[0],bead.center[1]),1,(0,0,255),1)

		cv2.imwrite(os.path.join(output_path, 'cycle_%i.png' %i), cimg) 




def get_cross_section_profile(image_manager, num_cycles, output_path):

	r = int(np.ceil(num_cycles/4))
	fig, ax = plt.subplots(nrows=r, ncols=4, figsize=(40,35))
	colors = ["#9467bd", "#ff7f0e", "#7f7f7f", '#8c564b', '#2ca02c', '#d62728']

	for i, (cycle, ax1) in enumerate(zip(image_manager.cycles, ax.flatten())):
		channels = cycle['raw_channels']

		for j, channel in enumerate(channels):
			w, h = channels[0].shape

			start = (int(w/2), 0)
			end = (int(w/2), h)
			p = profile_line(channel, start, end, linewidth=200, mode='constant')
			ax1.plot(p, linewidth=2, color=colors[j])
			ax1.set_ylim([0, 100])
			ax1.set_xlabel('Position')
			ax1.set_ylabel('Intensity')
			ax1.set_title('Cycle %s' % str(i+1))
	fig.legend(['Ch1', 'Ch2', 'Ch3', 'Ch4'], loc="center left")
	plt.savefig(os.path.join(output_path, 'cross_section.png'),bbox_inches='tight',dpi=150)
	plt.close()

def save_detected_beads(image_manager, beads, output_path):
	output_path = os.path.join(output_path, "intermediate_files")
	if os.path.exists(output_path) == False:
		os.makedirs(output_path)

	cimg = cv2.cvtColor(image_manager._read_image(-1)[1],cv2.COLOR_GRAY2BGR)

	for bead in beads:
		cv2.circle(cimg,(bead.center[0],bead.center[1]),1,(0,0,255),1)
		cv2.circle(cimg,(bead.center[0],bead.center[1]),bead.radius,(0,255,0),1)

	cv2.imwrite(os.path.join(output_path, 'detected_beads.png'), cimg) 