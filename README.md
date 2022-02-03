# Optocoder
Optocoder is a computational framework that processes microscopy images to decode bead barcodes in space. It efficiently aligns images, detects beads, and corrects for confounding factors of the fluorescence signal, such as crosstalk and phasing. Furthermore, Optocoder employs supervised machine learning to strongly increase the number of matches between optically decoded and sequenced barcodes

## Input
A yaml file can be used to specify the parameters of the run. The config should include the following information.

> **image_folder_path:** This is the path of the folder with the microscopy images. Images are expected to be multichannel tiff files.

> **file_name_seperator_loc:** Image files are assumed to have names splitted with underscore (e.g P1_10th_cycle_MMStack_Pos0.ome.tif). This seperator specifies the locaton of the cycle id so that the cycles can be correctly sorted.

> **num_cycles:** Number of imaging cycles (i.e number of images)

> **puck_id:** ID or name of the puck

> **output_path:** This is the path of the folder to save the results.

> **illumina_path:** If there is a corresponding illumina run available, this is the path of the txt file with cell barcodes.

> **channels_to_use:** This specifies the channels to use if cycle images have more than 4 channels
```
  - 0
  - 1
  - 4
  - 5
```

> **nuc_order:** This specifies the order of the bases to use if cycle images have more than 4 channels
```
  - 'T'
  - 'G'
  - 'C'
  - 'A'
```
Optionally, if the input comes from a SOLiD sequencing sample such as Slide-Seq v1 pucks, following information should be added:

> is_solid: True

> lig_seq: [1, 6, 0, 5, 4, 3, 2] # ligation sequence for the sample

> nuc_seq: [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 15, 17, 19] # cycle ids to use

## Running
Basic Optocoder module for a config file (i.e test_config.yaml) can be run with:
```python3 -m optocoder -config test_config.yaml```

If there is a Illumina barcode file is available, and we want to test phasing/prephasing parameters:
```python3 -m optocoder -config test_config.yaml --test_phasing```

If there is a Illumina barcode file is available, and we want to run machine learning module:
```python3 -m optocoder -config test_config.yaml --run_ml```

These options can also be combined to run everything at once:
```python3 -m optocoder -config test_config.yaml --test_phasing --run_ml```