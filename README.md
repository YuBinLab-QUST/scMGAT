# scMGAT

scMGAT: Improving single-cell multi-omics data analysis based on graph attention networks




## Related environment installation of scGMAT


- Please install the requirements (listed in environment.yaml). We're using Anaconda3 to install the environment:
```
conda create --name scMGAT python = 3.6
conda activate scMGAT
conda create -f environment.yaml

pip install numpy == 1.19.5
pip install pandas == 1.1.5
pip install dgl == 0.7.1
pip install tqdm == 4.50.0
pip install scikit-learn == 0.24.0
pip install matplotlib == 3.3.2
pip install torch == 1.8.1
pip install torch-geometric == 2.0.2 
pip install torch-cluster == 1.5.9 (https://pytorch-geometric.com/whl/)
pip install torch-scatter == 2.0.6 (https://pytorch-geometric.com/whl/)
pip install torch-sparse == 0.6.10 (https://pytorch-geometric.com/whl/)
pip install torch-spline-conv == 1.2.1 (https://pytorch-geometric.com/whl/)
```


## Guiding Principles:

We provide a single-cell multi-omics dataset from the mouse brain as an example, with the preprocessed dataset in the folder. The dataset we used is available from Gene Expression Omnibus (GEO) repository number under accession: GSE140203, and the sample code for this example is GSM4156599. Other datasets are available from the corresponding numbers given in the paper.

## Usage:

### Input
The input of scMGAT should be a csv file (row: cells, col: genes). You need the same number of cells and the same order of names for both omics, and the real cell labels need to be represented by characters.

### Run

Run the `train.py` to train the data， the `data` is the single-cell multi-omics matrix. 

### out

- `encodings.csv` The low-dimensional embedded data.
- `outputs_jieguo.csv` The single-cell multi-omics data for combined analysis.


## NOTICE

In the process of using, you need to adjust the data to the form of example data, and then store it in the same folder as `train.py`. It is worth noting that `train_gat.py` only has the graph attention network module, and its running steps are the same as above.



