import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24, 'lines.linewidth':4})

def calculate_ssim(image_manager, num_cycles, output_path):
	# calculates registration score using the structural similarity metric
	unregistered_ssim = image_manager.unreg_similarity_to_ref
	registered_ssim = image_manager.reg_similarity_to_ref
	
	ssims = pd.DataFrame(np.column_stack([unregistered_ssim, registered_ssim]), 
                               columns=['Unregistered Score', 'Registered Score'])

	ssims.to_csv(os.path.join(output_path, 'ssim.csv'))
	
	ax = ssims.plot(kind='line', marker='o',figsize=(16,10),  ms=18)
	plt.title("Structural Similarity Score")
	plt.xlabel("Cycles")
	plt.ylabel("Score")
	plt.axhline(y=0.5, color='r', linestyle='--', lw=2, label='threshold')
	plt.xticks(np.arange(0, num_cycles), np.arange(1,num_cycles+1))
	labels = ["Unregistered Score", "Registered Score", "Threshold"]
	handles, _ = ax.get_legend_handles_labels()

	plt.legend(handles = handles, labels = labels)

	plt.savefig(os.path.join(output_path, 'reg_score.png'),bbox_inches='tight', dpi=150)
	plt.close()