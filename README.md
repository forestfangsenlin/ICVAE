# ICVAE
ICVAE: Interpretable conditional variational autoencoder for de novo molecular design

<img src="https://github.com/forestspike/ICVAE/blob/main/image/ICVAE.jpg" width="800" />

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

You should also download the **normalized molecular property labels** and **origin property labels** from the "prop_np" fold.  

The "prop_np" file contains seven subfolds: molecular weight (weight), logP, sas, tpsa, qed, hba, hbd. In each fold,  the **y_train_norm.npy** and **y_test_norm.npy** files is the normalized molecular property value ranging from 0 to 500. We set the big normalized range to make the latent value of each property has some distance, which make the sampling process more easily to generate  smiles.

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
For training each molecular property, you can train ICVAE model by running **train.ipynb**.

### 4. generate the following latent image

<img src="https://github.com/forestspike/ICVAE/blob/main/image/MW_latent.jpg" width="600" />

you can just run the **plot_latent.ipynb** to obtain the molecular latent image.

### 5. sample the molecular with given the condition latent input

you can run the **sampling.ipynb** to generate the molecule with target property.

## Issues
Please report all installation / usage issues by opening an [issue](https://github.com/forestspike/ICVAE/issues) at this repo.

## References
Portions of the code have been re-used from the following repositories:
 * [topazape/molecular-VAE](https://github.com/topazape/molecular-VAE)
 * [jaechanglim/CVAE](https://github.com/jaechanglim/CVAE)
