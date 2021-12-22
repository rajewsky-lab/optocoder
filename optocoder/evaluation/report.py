import fpdf
import os
import pandas as pd
import yaml

def create_report(output_orig, output_plot_path, report_name):
    with open(os.path.join(output_orig, 'report_summary.yaml'), 'r') as infile:
        report_summary = yaml.load(infile)

    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(90, 10, " ", 0, 1, 'C')
    pdf.cell(90, 10, " ", 0, 1, 'C')
    pdf.cell(10)
    pdf.cell(10)
    pdf.cell(100, 8, "Optical Sequencing QC v."+"0.1.4"+ 
        ", enes.senel@mdc-berlin.de", 0, 1, 'L')
    pdf.cell(90, 8, " ", 0, 1, 'C')
    pdf.cell(10)
    pdf.cell(40, 8, 'Puck ID', 1, 0, 'C')
    pdf.cell(40, 8, '# Cycles', 1, 0, 'C')
    pdf.cell(40, 8, '# Detected Beads', 1, 1, 'C')
    pdf.cell(10)
    pdf.cell(40, 8, report_summary['puck_id'], 1, 0, 'C')
    pdf.cell(40, 8, format(report_summary['num_cycles'], ','), 1, 0, 'C')
    pdf.cell(40, 8, format(report_summary['num_beads'], ','), 1, 1, 'C')

    pdf.cell(90, 8, " ", 0, 1, 'C')
    pdf.cell(10)
    pdf.cell(40, 8, "Basecalling Method", 1, 0, 'C')
    pdf.cell(40, 8, '# Unique Barcodes', 1, 0, 'C')

    pdf.ln(h='')
    pdf.cell(10)
    pdf.cell(60, 8, "Naive", 1, 0, 'C')
    pdf.cell(60, 8, format(report_summary['unique_barcodes']['naive'], ','), 1, 0, 'C')
    pdf.ln(h='')
    pdf.cell(10)
    pdf.cell(60, 8, "Only Crosstalk Correction", 1, 0, 'C')
    pdf.cell(60, 8, format(report_summary['unique_barcodes']['only_ct'], ','), 1, 0, 'C')
    pdf.ln(h='')
    pdf.cell(10)
    pdf.cell(60, 8, "Crosstalk + Phasing Correction", 1, 0, 'C')
    pdf.cell(60, 8, format(report_summary['unique_barcodes']['phasing'], ','), 1, 0, 'C')

    pdf.ln(h='')
    pdf.cell(90, 20, " ", 0, 1, 'C')
    pdf.cell(80, 8, "Registration Score (Higher is better)", 1, 0, 'C')
    pdf.cell(15)
    pdf.cell(80, 8, "Average intensities through cycles", 1, 0, 'C')

    pdf.image(os.path.join(output_plot_path, 'reg_score.png'), x=10, y=125, w=80, h=50, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'average_intensities.png'), x=110, y=125, w=80, h=50, type='', link='')
    
    pdf.cell(90, 70, " ", 0, 1, 'C')

    pdf.cell(100, 8, "Basecalling fractions of the barcodes", 1, 1, 'C')
    pdf.image(os.path.join(output_plot_path, 'fractions_naive.png'), x=10, y=200, w=55, h=40, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'fractions_only_ct.png'), x=75, y=200, w=55, h=40, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'fractions_phasing.png'), x=140, y=200, w=55, h=40, type='', link='')

    pdf.ln(h='')
    pdf.cell(90, 50, " ", 0, 1, 'C')
    pdf.add_page()

    pdf.cell(100, 8, "Entropy of barcodes (higher values are better)", 1, 1, 'C')
    pdf.image(os.path.join(output_plot_path, 'entropy_histogram_naive.png'), x=30, y=20, w=60, h=50, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'entropy_histogram_only_ct.png'), x=95, y=20, w=60, h=50, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'entropy_histogram_phasing.png'), x=30, y=75, w=60, h=50, type='', link='')

    pdf.cell(125, 115, " ", 0, 1, 'C')

    pdf.cell(100, 8, "Compression length of barcodes (higher values are better)", 1, 1, 'C')
    pdf.image(os.path.join(output_plot_path, 'compression_histogram_naive.png'), x=30, y=150, w=55, h=40, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'compression_histogram_only_ct.png'), x=95, y=150, w=55, h=40, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'compression_histogram_phasing.png'), x=30, y=210, w=55, h=40, type='', link='')

    pdf.cell(125, 60, " ", 0, 1, 'C')
    pdf.add_page()

    pdf.cell(100, 8, "Confidences scores (higher values are better)", 1, 1, 'C')
    pdf.image(os.path.join(output_plot_path, 'confidence_joyplot_naive.png'), x=30, y=20, w=60, h=60, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'confidence_joyplot_only_ct.png'), x=95, y=20, w=60, h=60, type='', link='')
    pdf.image(os.path.join(output_plot_path, 'confidence_joyplot_phasing.png'), x=30, y=90, w=60, h=60, type='', link='')

    pdf.cell(125, 135, " ", 0, 1, 'C')

    pdf.cell(100, 8, "Intensity profile of the puck's central cross section", 1, 1, 'C')

    pdf.image(os.path.join(output_plot_path, 'cross_section.png'), x=None, y=None, w=150, h=110, type='', link='')
    pdf.add_page()

    pdf.cell(100, 8, "Raw intensities of random beads", 1, 1, 'C')
    pdf.image(os.path.join(output_plot_path, 'intensities_random_beads.png'), x=None, y=None, w=180, h=130, type='', link='')
    pdf.add_page()
    pdf.cell(120, 8, "Basecall scores of random beads (naive)", 1, 1, 'C')

    pdf.image(os.path.join(output_plot_path, 'random_beads_naive.png'), x=None, y=None, w=180, h=130, type='', link='')
    pdf.add_page()
    pdf.cell(120, 8, "Basecall scores of random beads (crosstalk correction)", 1, 1, 'C')

    pdf.image(os.path.join(output_plot_path, 'random_beads_only_ct.png'), x=None, y=None, w=180, h=130, type='', link='')
    pdf.add_page()
    pdf.cell(120, 8, "Basecall scores of random beads (crosstalk + phasing correction)", 1, 1, 'C')

    pdf.image(os.path.join(output_plot_path, 'random_beads_phasing.png'), x=None, y=None, w=180, h=130, type='', link='')

    pdf.output(report_name, 'F') 