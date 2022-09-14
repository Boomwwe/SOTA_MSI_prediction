# SOTA_MSI_prediction

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

We developed an efficient workflow for biomarkers in CRC (MSI, hypermutation, chromosomal
instability, CpG island methylator phenotype, BRAF, and TP53 mutation) that required relatively small datasets, but achieved
a state-of-the-art (SOTA) predictive performance.
![image](https://github.com/Boomwwe/SOTA_MSI_prediction/blob/main/MSI_code/Figure1.png)
## Table of Contents

- [Extract tiles](#extract_tiles)
- [Pre-training model](#pre-training_model)
- 
## Extract_tiles

The orginal code of this step is from [kather lab](https://github.com/KatherLab/preProcessing). And we modify it to generate tiles more easily.
```sh
$ python extractTiles.py -s slide_path -o out_path -ps pic_save_path
```
## Pre-training_model
A tiny Swin-T model was pre-trained to develop a multiclass tissue classifier. The tissue classifier was trained and tested using two publicly available pathologist-annotated datasets (NCT-CRC-HE-100K and CRC-VAL-HE-7K) from [Kather et al.](https://zenodo.org/record/1214456). These datasets consist of CRC image tiles of nine tissue types: adipose tissue (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), and colorectal adenocarcinoma epithelium (TUM)
```sh
$ python Pretrain.py -tr train_dir -te test_dir -sp save_path
```
