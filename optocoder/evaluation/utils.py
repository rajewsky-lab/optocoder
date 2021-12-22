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

def random_dist(length):
    # create random distrubution of barcodes
	return [randomword(12) for _ in range(length)]

def compress_string(barcode):
    # calculate the compressed barcode with string compression
	return ''.join(letter+str(len(list(group))) for letter, group in groupby(barcode))
