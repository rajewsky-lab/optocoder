import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import quantile_transform, scale
import cv2

def sample(component_1):
	"""Bin and sample beads for the selected component. 
	Basic idea is to choose beads in a channel pair to be able to 
	calculate the slope of the arm for a channel.
	"""
	component_1.columns = ["intensity", "intensity_comp2"]

	#calculate values for 60-99 quantiles
	component_1_upper_q = component_1.intensity.quantile(.99)
	component_1_lower_q = component_1.intensity.quantile(.60)
	
	#get the filtered beads
	component_1 = component_1[component_1.intensity < component_1_upper_q]
	component_1 = component_1[component_1.intensity > component_1_lower_q]
	component_1 = component_1.dropna()

	#bin the intensities
	#bins = np.linspace(component_1.intensity.min(), component_1.intensity.max(), 10000)
	
	#groups = component_1.groupby(pd.cut(component_1.intensity, bins))
	groups = component_1.groupby(np.arange(len(component_1.index))//10,axis=0)
	final_beads = pd.DataFrame()
	#get the extreme values
	for key, group in groups:
		min_row = group[group.intensity_comp2 == group.intensity_comp2.min()]
		final_beads = final_beads.append(pd.DataFrame(min_row))

	return final_beads

def regress(data):
	"""Get the slope with linear regression"""
	x = data["intensity"]
	y = data["intensity_comp2"]
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	return slope

def apply_w(vals, w_):
	#Apply the crosstalk matrix to transform the data
	vals = np.array(vals).T
	w = np.linalg.inv(w_)
	new_vals = np.matmul(w, vals)
	return pd.Series(new_vals)

def update(df, w_):
	df[["ch1","ch2","ch3","ch4"]] = df.apply(lambda x: apply_w([x.ch1, x.ch2,x.ch3,x.ch4], w_), axis=1)
	return df

def calculate_crosstalk_matrix(beads):

	#setup intensity dataset
	intensities = [np.asarray(bead.intensities['bc'][0])[[0,1,2,3]] for bead in beads]
	intensities = np.vstack(intensities).astype(int) 
	intensities = pd.DataFrame(intensities, columns = ['ch1', 'ch2', 'ch3', 'ch4'])

	num_iterations = 15
	threshold = 0.05

	#initial weight matrix
	w = np.identity(4) 
	dataset = intensities

	for iteration in range(num_iterations):
		#get the channel pairs
		pairs = []
		for i in [1,2,3,4]:
			for j in [1,2,3,4]:
				ch = dataset[["ch%i"%i, "ch%i"%j]]
				pairs.append(ch)

		w_ = []
		#for every pair, calculate the slope
		for pair in pairs:
			data = sample(pair)
			slope = regress(data)
			w_.append(slope)

		w_ = np.reshape(np.array(w_), (4,4)).T

		#check the non diagonal elements and end if the max slope is lower than the threshold
		non_diag = w_[~np.eye(w_.shape[0],dtype=bool)].reshape(w_.shape[0],-1)
		max_slope = (np.max(np.abs(non_diag)))
		if max_slope <= threshold:
			break

		#update the dataset and the crosstalk matrix
		dataset = update(dataset, w_)
		w = np.matmul(w,w_)

	#normalize the crosstalk matrix
	w_norm = w/w.sum(axis=0,keepdims=1)
	
	return w_norm