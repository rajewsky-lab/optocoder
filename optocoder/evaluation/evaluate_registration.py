import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def calculate_ssim(image_manager, num_cycles, output_path):

	# calculates registration score using the structural similarity metric
	unregistered_ssim = image_manager.unreg_similarity_to_ref
	registered_ssim = image_manager.reg_similarity_to_ref
	
	ssims = pd.DataFrame(np.column_stack([unregistered_ssim, registered_ssim]), 
                               columns=['Unregistered Score', 'Registered Score'])

	ssims.to_csv(os.path.join(output_path, 'ssim.csv'))
	tick_font_params = dict(tickfont = dict(family="Arial",size=70))

	colors = px.colors.qualitative.D3
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1), y=ssims['Unregistered Score'],marker=dict(size=20),line=dict(color=colors[1], width=15),
                    name='Unregistered Scores', mode='lines+markers'))
	fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1),y=ssims['Registered Score'],marker=dict(size=20),line=dict(color=colors[0], width=15),
                    name='Registered Scores', mode='lines+markers'))
	fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1),y=np.repeat(0.5, num_cycles),line=dict(dash='dashdot', color='red', width=4),
                    name='Threshold', mode='lines'))

	fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
	fig.update_xaxes(tickmode='linear', showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

	fig.update_layout(
    xaxis_title="Cycles",
    yaxis_title="Registration Score",
    font=dict(
        family="Arial",
        size=70,
    ))
	fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=0.75
	))
	fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
	fig.update_layout(legend_itemsizing='trace')
	fig.update_layout(legend_itemwidth=120)
	fig.write_image(os.path.join(output_path, 'reg_score.png'), scale=2)
	fig.write_image(os.path.join(output_path, 'svgs', 'reg_score.svg'), scale=2)
