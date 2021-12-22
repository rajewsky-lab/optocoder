import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import zip_longest, cycle, groupby
import pandas as pd
from math import log, e
import joypy
from matplotlib import cm
from matplotlib_venn import venn3
from collections import Counter
import random, string
import cv2
import seaborn as sns
from optocoder.evaluation.utils import freq, calculate_entropy, random_dist, randomword, compress_string

plt.rcParams.update({'font.size': 24, 'lines.linewidth':5})
plt.rcParams["font.family"] = "Arial"

def evaluate_fractions(beads, num_cycles, output_path, method):
	# calculate the nucleotide fractions for the cycles

	barcodes = [bead.barcode[method] for bead in beads]
	fractions = [freq(code) for code in zip_longest(*barcodes)]

	ax = pd.DataFrame(fractions).plot(kind='line', figsize=(16,10), marker='o')
	plt.ylabel("Base Fractions")
	plt.xlabel("Cycles")
	plt.title('Base Fractions (%s)' % method)
	ax.set_xticks(np.arange(0, num_cycles))
	ax.set_xticklabels(np.arange(1,num_cycles+1))
	ax.set_ylim([0,0.8])
	plt.savefig(os.path.join(output_path, 'fractions_%s.png' % method),bbox_inches='tight', dpi=150)
	plt.close()

def plot_barcode_entropy(beads, output_path, method):

	ents = []
	for bead in beads:
		entropy = calculate_entropy(bead.barcode[method])
		ents.append(entropy)

	rand_ints = []
	for r in random_dist(len(beads)):
		rand_ints.append(calculate_entropy(r))

	plt.figure(figsize=(12,8))
	plt.hist(np.array(ents), density=False, bins=40, alpha=0.5, label='barcodes')
	plt.hist(np.array(rand_ints), density=False, bins=40, alpha=0.5, label='theoretical')

	plt.title('Entropy Score (%s)' % method)
	plt.ylabel('Count')
	plt.xlabel('Entropy Value')
	plt.legend(loc='upper left')
	
	plt.savefig(os.path.join(output_path, 'entropy_histogram_%s.png' % method),bbox_inches='tight', dpi=150)
	plt.close()

def plot_barcode_compression(beads, output_path, method):
	complexities = []
	for bead in beads:
		complexity = len(compress_string(bead.barcode[method]))
		complexities.append(complexity)

	rand_ints = []

	for r in random_dist(len(beads)):
		rand_ints.append(len(compress_string(r)))

	plt.figure(figsize=(12,8))
	plt.hist(np.array(complexities), density=False, bins=40, alpha=0.5, label='barcodes')
	plt.hist(np.array(rand_ints), density=False, bins=40, alpha=0.5, label='theoretical')

	plt.title('Compression Score (%s)' % method)
	plt.ylabel('Count')
	plt.xlabel('Compressed length')
	plt.legend(loc='upper left')

	plt.savefig(os.path.join(output_path, 'compression_histogram_%s.png' % method),bbox_inches='tight', dpi=150)
	plt.close()

def plot_barcode_confidences(beads, num_cycles, output_path, method):

	confidences = [np.max(np.vstack(bead.scores[method]), axis=1) for bead in beads]

	labels=['Cycle %i' % i for i in np.arange(1,num_cycles+1)]
	fig, axes = joypy.joyplot(pd.DataFrame(np.matrix(confidences), columns=labels), fade=True, colormap=cm.autumn_r, figsize=(12, 8))
	plt.title('Basecall scores (%s)' % method)
	plt.xlabel('Confidence Score')
	plt.ylabel('Cycles')
	plt.savefig(os.path.join(output_path, 'confidence_joyplot_%s.png' % method),bbox_inches='tight', dpi=150)
	plt.close()

def plot_selected(beads, output_path):
	# plot intensity profiles of randomy selected beads
	import random
	bead = random.choice(beads)
	plt.figure()
	plt.plot([1,2,3,4,5,6],bead.intensities[0]/np.max(bead.intensities[0]), linestyle='-', marker='o')
	plt.xlabel('Channel', fontsize=24)
	plt.ylabel('Normalised Intensity', fontsize=24)
	
	plt.savefig(os.path.join(output_path, 'selected_beads.png'),bbox_inches='tight', dpi=150)
	plt.close()

def plot_cycle_scores(beads, method, num_cycles, output_path):
	confidences = [bead.scores[method] for bead in beads]
	fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(40,40))

	i = 0
	for row in ax:
		for col in row:
			a = pd.DataFrame(confidences[i]).plot(kind='bar', ax=col, legend=False)
			a.set_title('ID: %s' % str(beads[i].id))
			a.set_ylim([0, 1])
			a.set_xticks(np.arange(0, num_cycles))
			a.set_xticklabels(np.arange(1,num_cycles+1))
			a.set_xlabel('Cycles')
			a.set_ylabel('Score')
			i = i+1
	fig.legend(['G', 'T', 'A', 'C'], loc="center left")
	plt.savefig(os.path.join(output_path, 'random_beads_%s.png' % method),bbox_inches='tight',dpi=150)
	plt.close()

def evaluate_cycles(beads, num_cycles, output_path, methods):
	
	beads = np.random.choice(beads, 16)
	
	for method in methods: plot_cycle_scores(beads, method, num_cycles, output_path) 

	fig, ax = plt.subplots(nrows=num_cycles, ncols=5, figsize=(40,40))
	intensities = [bead.intensities['raw'] for bead in beads[:5]]

	i = 0
	j = 0

	for row in ax:
		i = 0
		if num_cycles == 1:
			row = [row]

		for col in row:
			a = pd.DataFrame(intensities[i][j]).plot(kind='line', ax=col, legend=False)
			if j == 0:
				a.set_title('ID: %s' % beads[i].id)
			a.set_xticks(np.arange(0, 4))
			a.set_xticklabels(np.arange(1,5))
			
			a.set_xlabel('Channels')
			a.set_ylabel('Intensity')
			i = i+1

		if num_cycles == 1:
			break
		j = j + 1

	plt.savefig(os.path.join(output_path, 'intensities_random_beads.png'),bbox_inches='tight',dpi=150)
	plt.close()




def output_prediction_data(beads, num_cycles, output_path, method):
	# save barcodes and scores to csv

	bead_ids = [bead.id for bead in beads]
	locxs = [bead.center[0] for bead in beads]
	locys = [bead.center[1] for bead in beads]

	barcodes = [bead.barcode[method] for bead in beads]

	d = {"bead_id": bead_ids, "x_pos": locxs, "y_pos": locys, "barcodes": barcodes}
	df = pd.DataFrame(d)

	confidences = [bead.scores[method] for bead in beads]
	confidences = [np.hstack(c) for c in confidences]

	df_scores = pd.DataFrame(confidences, columns=['score_cycle_%i_nuc_%s' % (i,j) for i in range(1,num_cycles+1) for j in ['G', 'T', 'A', 'C']])
	
	df = pd.concat([df, df_scores], axis=1)

	df.to_csv(os.path.join(output_path,'predictions_%s.csv' % method), index=False, sep='\t')

def save_phasing_plot(values, output_path):
	plt.rcParams.update({'font.size': 24, 'lines.linewidth':5})
	plt.rcParams["font.family"] = "Arial"

	ser = pd.Series([v[0] for v in list(values.values())],
					index=pd.MultiIndex.from_tuples(values.keys()))
	df = ser.unstack().fillna(0)

	ser = pd.Series([v[1] for v in list(values.values())],
					index=pd.MultiIndex.from_tuples(values.keys()))
	df2 = ser.unstack().fillna(0)

	vmin = min(df.values.min(), df2.values.min())
	vmax = max(df.values.max(), df2.values.max())

	fig, axs = plt.subplots(figsize=(16,8), ncols=3, gridspec_kw=dict(width_ratios=[4,4,0.2]))

	h1 = sns.heatmap(df, annot=True, cbar=False, ax=axs[0], vmin=vmin, fmt='g')
	h2 = sns.heatmap(df2, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax, fmt='g')

	h1.set_xticklabels(h1.get_xticklabels(), rotation = 0)
	h2.set_xticklabels(h1.get_xticklabels(), rotation = 0)
	axs[0].title.set_text('Number of Matches')
	axs[1].title.set_text('Number of Neg Ctrl Matches')

	fig.text(0.5, 0.02, 'Prephasing Probability', ha='center')
	fig.text(0.07, 0.5, 'Phasing Probability', va='center', rotation='vertical')
	fig.colorbar(axs[1].collections[0], cax=axs[2])
	plt.savefig(os.path.join(output_path, 'phasing_grid.png'),bbox_inches='tight',dpi=150)
	plt.close()