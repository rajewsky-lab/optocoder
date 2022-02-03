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
from optocoder.evaluation.utils import freq, calculate_entropy, random_dist, randomword, compress_string, get_cycle_scores
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from plotly import tools
from plotly.subplots import make_subplots
def evaluate_fractions(beads, num_cycles, output_path, method):
    tick_font_params = dict(tickfont = dict(family="Arial",size=70))

    # calculate the nucleotide fractions for the cycles
    barcodes = [bead.barcode[method] for bead in beads]
    fractions = [freq(code) for code in zip_longest(*barcodes)]
    fractions = pd.DataFrame(fractions)
    colors = px.colors.qualitative.D3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1), y=fractions['G'],marker=dict(size=20),line=dict(color=colors[0], width=15),
                    name='G', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1),y=fractions['T'],marker=dict(size=20),line=dict(color=colors[1], width=15),
                    name='T', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1),y=fractions['A'],marker=dict(size=20),line=dict(color=colors[2], width=15),
                    name='A', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=np.arange(1,num_cycles+1),y=fractions['C'],marker=dict(size=20), line=dict(color=colors[3], width=15),
                    name='C', mode='lines+markers'))
    
    fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
    fig.update_layout(legend_itemsizing='trace')
    fig.update_layout(legend_itemwidth=120)

    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(tickmode='linear', showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

    fig.update_layout(
    xaxis_title="Cycles",
    yaxis_title=f"Base Fractions ({method})",
    font=dict(
        family="Arial",
        size=70,
    ))
    fig.update_layout(yaxis_range=[0,0.8])
    fig.write_image(os.path.join(output_path, f'fractions_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'fractions_{method}.svg'), scale=2)

def plot_barcode_entropy(beads, num_cycles, output_path, method):
    tick_font_params = dict(tickfont = dict(family="Arial",size=70))

    ents = []
    for bead in beads:
        entropy = calculate_entropy(bead.barcode[method])
        ents.append(entropy)

    rand_ints = []
    
    for r in random_dist(len(beads), num_cycles):
        rand_ints.append(calculate_entropy(r))

    fig = go.Figure()
    entropy_data = pd.DataFrame()
    entropy_data['barcodes'] = np.array(ents)
    entropy_data['theoretical'] = np.array(rand_ints)
    trace0 = go.Histogram(
        x=entropy_data['barcodes'],
        name='Barcodes',nbinsx=40,
        marker={'color':'#1f77b4'},
    )
    trace1 = go.Histogram(
        x=entropy_data['theoretical'],
        name='Theoretical', nbinsx=40,
        marker={'color':'#ff7f0e'}
    )
    fig.add_trace(trace0)
    fig.add_trace(trace1)
    fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
    fig.update_layout(legend_itemsizing='trace')
    fig.update_layout(legend_itemwidth=120)

    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(tickmode='linear', showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

    fig.update_layout(
    xaxis_title=f"Entropy ({method})",
    yaxis_title="Counts",
    font=dict(
        family="Arial",
        size=70,
    ))
    fig.write_image(os.path.join(output_path, f'entropy_histogram_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'entropy_histogram_{method}.svg'), scale=2)

def plot_barcode_compression(beads, num_cycles, output_path, method):
    tick_font_params = dict(tickfont = dict(family="Arial",size=70))

    complexities = []
    for bead in beads:
        complexity = len(compress_string(bead.barcode[method]))
        complexities.append(complexity)

    rand_ints = []

    for r in random_dist(len(beads), num_cycles):
        rand_ints.append(len(compress_string(r)))

    fig = go.Figure()
    entropy_data = pd.DataFrame()
    entropy_data['barcodes'] = np.array(complexities)
    entropy_data['theoretical'] = np.array(rand_ints)
    trace0 = go.Histogram(
        x=entropy_data['barcodes'],
        name='Barcodes',
        marker={'color':'#1f77b4'},
    )
    trace1 = go.Histogram(
        x=entropy_data['theoretical'],
        name='Theoretical',
        marker={'color':'#ff7f0e'}
    )
    fig.add_trace(trace0)
    fig.add_trace(trace1)
    fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
    fig.update_layout(legend_itemsizing='trace')
    fig.update_layout(legend_itemwidth=120)

    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

    fig.update_layout(
    xaxis_title=f"Compression Score ({method})",
    yaxis_title="Counts",
    font=dict(
        family="Arial",
        size=70,
    ))
    fig.write_image(os.path.join(output_path, f'compression_histogram_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'compression_histogram_{method}.svg'), scale=2)

def plot_barcode_confidences(beads, num_cycles, output_path, method):
    from plotly.colors import n_colors

    confidences = [np.max(np.vstack(bead.scores[method]), axis=1) for bead in beads]
    confidences = pd.DataFrame(np.matrix(confidences))
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', num_cycles, colortype='rgb')
    fig = go.Figure()
    for i, (data_line, color) in enumerate(zip(reversed(confidences.columns), reversed(colors))):
        fig.add_trace(go.Violin(x=confidences[data_line], line_color=color, name=f'Cycle {num_cycles-i}'))
    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    fig.update_layout(  width=1700,
    height=1100,
    paper_bgcolor='white',
    plot_bgcolor='white')
    tick_font_params = dict(tickfont = dict(family="Arial",size=70))

    fig.update_yaxes(tickmode='linear', tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(tickfont=tick_font_params['tickfont'])

    fig.update_layout(
    xaxis_title="Chastity Score (%s)" % method,
    yaxis_title="Cycles",
    font=dict(
        family="Arial",
        size=70,
    ))
    fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
    fig.update_layout(legend_itemsizing='trace')
    fig.update_layout(legend_itemwidth=120)
    fig.update_layout(showlegend=False)

    fig.write_image(os.path.join(output_path, f'confidence_joyplot_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'confidence_joyplot_{method}.svg'), scale=2)

def plot_cycle_scores(beads, method, num_cycles, output_path):

    confidences = [np.max(np.vstack(bead.scores[method]), axis=1) for bead in beads]
    confidences_bases = [np.argmax(np.vstack(bead.scores[method]), axis=1) for bead in beads]

    confidences = pd.DataFrame(np.matrix(confidences))
    confidences_bases = pd.DataFrame(np.matrix(confidences_bases))
    colors = px.colors.qualitative.D3
    tick_font_params = dict(tickfont = dict(family="Arial",size=70))

    fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=([f'Bead ID: {b.id}' for b in beads]))
    i = 0
    for row in range(1,6):
        for col in range(1,3):

            fig.add_trace(go.Bar(x=np.arange(1, num_cycles+1), y=confidences.iloc[i,:], legendgroup='group1', showlegend=False, marker_color=[colors[x] for x in confidences_bases.iloc[i,:]]),
              row=row, col=col)
            fig.update_yaxes(title_text='Chastity Score', range=[0, 1.0], row=row, col=col)
            fig.update_xaxes(title_text="Cycles", row=row, col=col)
            i = i + 1
    fig.update_layout(
    width=5000,
    height=3000,
    paper_bgcolor='white',
    plot_bgcolor='white',
    )
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(tickmode='linear', tickangle=0, showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=50,
        )
    )
    fig.update_annotations(font_size=70)
    fig.update_layout(xaxis = tick_font_params, yaxis=tick_font_params)

    fig.write_image(os.path.join(output_path, f'random_beads_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'random_beads_{method}.svg'), scale=2)

def evaluate_cycles(beads, num_cycles, output_path, methods):

    beads = np.random.choice(beads, 10)
    
    for method in methods: plot_cycle_scores(beads, method, num_cycles, output_path) 

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

    plt.savefig(os.path.join(output_path, 'phasing_grid.png'),bbox_inches='tight',dpi=300)
    plt.close()

def plot_number_good_cycles(beads, num_cycles, output_path, method, threshold=0.85):

    optical_scores = get_cycle_scores(beads, num_cycles)

    bins = [0, threshold, 1.1]
    names = ['bad', 'good']
    df = pd.DataFrame(optical_scores.T)

    d = dict(enumerate(names, 1))
    for column in df:
        df[column] = np.vectorize(d.get)(np.digitize(df[column], bins))

    counts = df.apply(lambda x : x.value_counts() , axis = 1)[['good' , 'bad']]
    counts = counts.fillna(0)
    tick_font_params = dict(tickfont = dict(family="Arial",size=70))


    counts2 = df.T.apply(lambda x : x.value_counts() , axis = 1)[['good' , 'bad']]
    counts2 = counts2.fillna(0)
    plt.figure()
    fig = px.bar(counts2)
    
    fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
    fig.update_layout(legend_itemsizing='trace')
    fig.update_layout(legend_itemwidth=120)

    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(tickmode='array', tickvals =np.arange(0, num_cycles), ticktext=np.arange(1, num_cycles+1), showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

    fig.update_layout(
    xaxis_title="Cycles",
    yaxis_title="# of Beads",
    legend_title=f"Base Quality ({method})",
    font=dict(
        family="Arial",
        size=70,
    ))
    fig.write_image(os.path.join(output_path, f'locs_{threshold}_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'locs_{threshold}_{method}.svg'), scale=2)

    counts['good'] = counts['good'].astype(int)

    counts = counts['good'].value_counts()
    counts = counts.sort_index(ascending=True)
    fig = px.bar(counts)
    
    fig.update_layout(
    width=2200,
    height=1300,
    paper_bgcolor='white',
    plot_bgcolor='white')
    fig.update_layout(legend_itemsizing='trace')
    fig.update_layout(legend_itemwidth=120)

    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(tickmode='array', tickvals =np.arange(0, num_cycles), ticktext=np.arange(1, num_cycles+1), showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

    fig.update_layout(
    xaxis_title="Cycles",
    yaxis_title="# of Beads",
    legend_title=f"Base Quality ({method})",
    font=dict(
        family="Arial",
        size=70,
    ))
    fig.write_image(os.path.join(output_path, f'good_hist_{threshold}_{method}.png'), scale=2)
    fig.write_image(os.path.join(output_path, 'svgs', f'good_hist_{threshold}_{method}.svg'), scale=2)
