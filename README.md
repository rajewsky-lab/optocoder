# opseq
This package provides the optical sequencing pipeline for the Rajewsky lab spatial transcriptomics platform. 

## Input
Pipeline requires a yaml file as the config file for the run. 
```
--- 
 image_folder_path: "path/to/images"
 num_cycles: 12
 puck_id: 'pid_00000'
 output_path: "path/to/output"
 only_analysis: 0
 illumina_path: "path/to/illumina/topBarcodes.txt"
 machine_learning: 1
```
**Parameter descriptions:**
* **image_folder_path**: this is the path for the microscopy images (6ch x num of cycles tif files)
* **num_cycles**: number of cycles. this is 12 for a complete puck
* **output_path**: folder to save output files
* **only_analysis**: if this is 1, only the report files will be generated from the previous run
* **illumina_path**: path for the illumina barcodes, if exists
* **machine_learning**: if 1 and illumina barcodes are there, ml model will be trained for basecalling

***Important:*** Image folder path must include channel images with the format similar to 'PID_04th_cycle_B0003_1_MMStack_Pos0.ome.tif' where cycle id is seperated with an underscore.


## Example way to run
```
python3 -m opseq -config data/test.yaml
```

or for the ui:
```
python3 -m opseq -ui
```