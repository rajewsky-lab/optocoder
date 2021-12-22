import numpy as np
import itertools

def create_phasing_matrix(phasing_prob, prephasing_prob, num_cycles):
	"""Create a phasing matrix for the given phasing and prephasing probabilities

	Args:
		phasing_prob (float): phasing probability
		prephasing_prob (float): prephasing probability
		num_cycles (int): number of optical sequencing cycles

	Returns:
		ndarray: phasing matrix
	"""

	if num_cycles == 1:
		return np.ones((1,1))

	L = num_cycles
	phasing_matrix = np.zeros((L,L))

	phasing_matrix[0, 0] = 1 - prephasing_prob
	phasing_matrix[0, 1] = prephasing_prob

	for i,j in itertools.product(range(1,L), range(L)):
	    if j == 0:
	        phasing_matrix[i][j] = phasing_prob*phasing_matrix[i-1][j]
	    elif j == 1:
	        phasing_matrix[i][j] = phasing_prob*phasing_matrix[i-1][j] + (1-phasing_prob-prephasing_prob)*phasing_matrix[i-1][j-1]
	    else:
	        phasing_matrix[i][j] = phasing_prob*phasing_matrix[i-1][j] + (1-phasing_prob-prephasing_prob)*phasing_matrix[i-1][j-1] + prephasing_prob*phasing_matrix[i-1][j-2]

	return phasing_matrix

