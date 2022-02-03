from math import log, e
import numpy as np
import random, string
from itertools import zip_longest, cycle, groupby
from collections import Counter

def freq(code):
    # calculate the nucleotide frequencies in barcodes
	l = len(code) - code.count(None)
	return {n: code.count(n)/l for n in 'GTAC'}

def calculate_entropy(bead, base=2):
    # calculate shannon entropy
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

def randomword(length):
    # generate a random barcode
    letters = ['G','T','A','C']
    return ''.join(random.choice(letters) for i in range(length))

def random_dist(length, num_cycles):
    # create random distrubution of barcodes
	return [randomword(num_cycles) for _ in range(length)]

def compress_string(barcode):
    # calculate the compressed barcode with string compression
	return ''.join(letter+str(len(list(group))) for letter, group in groupby(barcode))

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