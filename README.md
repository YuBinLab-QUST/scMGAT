# scMGAT

scMGAT: Improving single-cell multi-omics data analysis based on graph attention networks


## scGMAI uses the following dependencies:

- `python = 3.6`
- `numpy = 1.19.5`
- `pandas = 1.1.5`
- `torch = 1.8.1`
- `dgl = 0.7.1`
- `tqdm = 4.50.0`
- `scikit-learn = 0.24.0`
- `matplotlib = 3.3.2`


## Guiding Principles:

We provide a single-cell multi-omics dataset from the mouse brain as an example, with the preprocessed dataset in the folder. The dataset we used is available from Gene Expression Omnibus (GEO) repository number under accession: GSE140203, and the sample code for this example is GSM4156599. Other datasets are available from the corresponding numbers given in the paper.

## Usage:

### Input
The input of scMGAT should be a csv file (row: cells, col: genes).

### Run

Run the `train.py` to train the data， the `data` is the single-cell multi-omics matrix.

### out

- `encodings.csv` The low-dimensional embedded data.
- `outputs_jieguo.csv` The single-cell multi-omics data for combined analysis.


## FUNDING

This work was supported by the National Natural Science Foundation of China (No. 62172248), the Natural Science Foundation of Shandong Province of China (No. ZR2021MF098). 
Conflict of interest statement. None declared.



