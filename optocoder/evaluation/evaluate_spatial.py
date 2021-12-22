import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24, 'lines.linewidth':4})
import matplotlib.patches as mpatches
from math import log, e
from itertools import zip_longest, cycle, groupby

def get_cycle_scores(optical, num_cycles):

	optical_scores = optical.iloc[:, 4:]
	optical_scores = np.array(optical_scores)
	optical_scores = np.hsplit(optical_scores, num_cycles)

	cycle_scores = []
	for cycle in optical_scores:
		maxes = np.max(cycle, axis=1)
		cycle_scores.append(maxes)

	cycle_scores = np.array(cycle_scores)
	return cycle_scores

def calculate_entropy(bead, base=2):

	labels = list(bead)

	if 'N' in labels:
		return 0
		
	n_labels = len(labels)

	if n_labels <= 1:
		return 0

	value,counts = np.unique(labels, return_counts=True)
	probs = counts / n_labels
	n_classes = np.count_nonzero(probs)

	if n_classes <= 1:
		return 0

	ent = 0.

	base = e if base is None else base
	for i in probs:
		ent -= i * log(i, base)

	return ent

def compress_string(barcode):
    return ''.join(letter+str(len(list(group))) for letter, group in groupby(barcode))

def map_color(nuc):
    mapper = {'G': (31/255.0, 119/255.0, 180/255.0), 'T': (255/255.0, 127/255.0, 14/255.0), 'A': (44/255.0, 160/255.0, 44/255.0), 'C': (214/255.0, 39/255.0,40/255.0 )} 
    return mapper[nuc]

def plot_barcodes_in_space(beads, num_cyles, output_path):

    output_path = os.path.join(output_path, "intermediate_files")
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    
    bcs = np.array(list(map(list, beads['barcodes'])))

    

    for cycle in range(num_cyles):
        plt.figure()
        plt.scatter(beads['x_pos'], beads['y_pos'], s = 0.1, c=list(map(map_color, bcs[:,cycle])))
        plt.axis('off')

        classes = ['G','T','A','C']
        class_colours = [(31/255.0, 119/255.0, 180/255.0),(255/255.0, 127/255.0, 14/255.0), (44/255.0, 160/255.0, 44/255.0), (214/255.0, 39/255.0,40/255.0 )]
        recs = []
        for i in range(0,len(class_colours)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
        plt.legend(recs,classes,loc='upper center', ncol=len(classes), bbox_to_anchor=(0.5, -0.05), prop={'size': 10})
        plt.savefig(os.path.join(output_path, f'cycle_{cycle}_bases_spatial.png'), bbox_inches='tight', dpi=150)
        plt.close('all')

def plot_entropy_in_space(beads, num_cyles, output_path):

    output_path = os.path.join(output_path, "intermediate_files")
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    
    bcs = np.array(list(map(list, beads['barcodes'])))

    
    ents = []
    for bead_barcode in beads['barcodes']:
        entropy = calculate_entropy(bead_barcode)
        ents.append(entropy)

    fig = plt.figure()
    sc = plt.scatter(beads['x_pos'], beads['y_pos'], s = 0.1, c=ents)
    plt.axis('off')
    cb_ax = fig.add_axes([.91,.124,.04,.754])
    cbar = fig.colorbar(sc,orientation='vertical',cax=cb_ax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(10)
    plt.savefig(os.path.join(output_path, f'entropy_spatial.png'), bbox_inches='tight', dpi=150)

def plot_compression_in_space(beads, num_cyles, output_path):

    output_path = os.path.join(output_path, "intermediate_files")
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    
    bcs = np.array(list(map(list, beads['barcodes'])))

    
    complexities = []
    for bead_barcode in beads['barcodes']:
        complexity = len(compress_string(bead_barcode))
        complexities.append(complexity)

    fig = plt.figure()
    sc = plt.scatter(beads['x_pos'], beads['y_pos'], s = 0.1, c=complexities, cmap='inferno')
    plt.axis('off')
    cb_ax = fig.add_axes([.91,.124,.04,.754])
    cbar = fig.colorbar(sc,orientation='vertical',cax=cb_ax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(10)
    plt.savefig(os.path.join(output_path, f'compression_spatial.png'), bbox_inches='tight', dpi=150)

def plot_chastity_in_space(beads, num_cycles, output_path):

    output_path = os.path.join(output_path, "intermediate_files")
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    
    bcs = np.array(list(map(list, beads['barcodes'])))

    puck_scores =  get_cycle_scores(beads, num_cycles)
    mean_scores = np.mean(puck_scores, axis=0)
    for cycle in range(num_cycles):
        fig = plt.figure()
        sc = plt.scatter(beads['x_pos'], beads['y_pos'], s = 0.1, c=puck_scores[cycle, :], cmap='plasma')
        plt.axis('off')
        cb_ax = fig.add_axes([.91,.124,.04,.754])
        cbar = fig.colorbar(sc,orientation='vertical',cax=cb_ax)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(10)
        plt.savefig(os.path.join(output_path, f'chastity_cycle+{cycle}.png'), bbox_inches='tight', dpi=150)
        plt.close('all')