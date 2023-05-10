# ICVAE
ICVAE: Interpretable conditional variational autoencoder for de novo molecular design

[![Python 3.6](https://img.shields.io/badge/python-3.6-yellow.svg)](https://www.python.org/downloads/release/python-367/)
[![License: GPL v3](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Environment
- python = 3.6.13
- pytorch = 1.10.1
- RDKit
- numpy
- matplotlib
- jupyter notebook

## How to run？
### 1. download dataset and save in your local path
Visit https://github.com/aksub99/molecular-vae/tree/master/ , and download the processed ZINC dataset from "data" fold. Then you can save it in your local path which contains the ICVAE code.

You should also download the normlized molecular property labels and origin labels from the "prop_np" fold.  

The "prop_np" contains seven subfolds: molecular weight (weight), logP, sas, tpsa, qed, hba, hbd. In each fold,  the **y_train_norm.npy** and **y_test_norm.npy** files is the normalized molecular property value ranging from 0 to 500. We set the big normalized range to make the latent value of each property has some distance, which make the sampling process more easily to generate  smiles.

### 2. set up the pytorch environment
We recommend you to install anaconda and create a new environment by using the following command:
```
conda create --name icvae python=3.6
```
and then you can enter the enviroment by (windows) :
```
conda activate icvae
```
or (ubuntu) :
```
source activate icvae
```
and next, you need to install the package by:

```
conda install numpy, matplotlib, jupyter notebook, rdkit -c rdkit
```

Please note your must install the gpu support for pytorch. The detail can be found in this blog: https://medium.com/analytics-vidhya/4-steps-to-install-anaconda-and-pytorch-onwindows-10-5c9cb0c80dfe .

### 3. launch the jupyter notebook and run the code

launch the jupyter notebook by:
```
jupyter notebook
```
Enter the ICVAE fold and The ICVAE contains two main models:

**ICVAE_1prop_gen.ipynb** is the ICVAE model that using one molecular property to train and can use the latent vaiable to control the generated molecular property. 

**ICVAE_2prop_gen.ipynb** is the ICVAE model that using two molecular property to train and can use the latent vaiable to control the two generated molecular properties.

For training each molecular property, you can just run after changing the condition input path into correspond molecular property.

## References
Portions of the code have been re-used from the following repositories:
 * [topazape/molecular-VAE](https://github.com/topazape/molecular-VAE)
 * [jaechanglim/CVAE](https://github.com/jaechanglim/CVAE)
