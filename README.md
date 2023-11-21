# SOTA_MSI_prediction

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

### We developed an efficient workflow for biomarkers in CRC (MSI, hypermutation, chromosomal instability, CpG island methylator phenotype, BRAF, and TP53 mutation) that required relatively small datasets, but achieved a state-of-the-art (SOTA) predictive performance.
![image](https://github.com/Boomwwe/SOTA_MSI_prediction/blob/main/MSI_code/Figure1.png)
## Table of Contents

- [Extract tiles](#extract_tiles)
- [Pre-training model](#pre-training_model)
- [Color normalization](#color_normalization)
- [Tumor selection](#tumor_selection)
- [Training model](#training_model)
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
## Color_normalization
The image tiles were color-normalized using Macenko’s method to reduce the color bias and improve classifier performance and were subsequently resized to 224×224 px to serve as the input of the network. The orginal code of this step is from [Li et al.](https://github.com/1996lixingyu1996/CRCNet)
```sh
$ python color_normalize.py -i input_dir -o output_dir
```

## Tumor_selection
The pre-trained tissue classifier was trained to detect and select tiles with tumor tissue.
```sh
$ python select_tumor.py -i input_dir -o output_dir -mp model_path
```

## Training_model
The pre-trained Swin-T model (tissue classifier) was fine-tuned for the binary classification of key CRC biomarkers at the patient (slide) level
```sh
$ python training.py -cv cv_dir -pp pic_dir -lp label_path -sp save_path
```

## Visualization
The interpretability of the Swin-T models was explored using visualization technology with Python package [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
![image](https://github.com/Boomwwe/SOTA_MSI_prediction/blob/main/MSI_code/Figure6(1).png)

## Citation
If you use this for research, please cite. Here is an example BibTeX entry:
```
@article{guo2023predicting,
  title={Predicting microsatellite instability and key biomarkers in colorectal cancer from H\&E-stained images: achieving state-of-the-art predictive performance with fewer data using Swin Transformer},
  author={Guo, Bangwei and Li, Xingyu and Yang, Miaomiao and Jonnagaddala, Jitendra and Zhang, Hong and Xu, Xu Steven},
  journal={The Journal of Pathology: Clinical Research},
  volume={9},
  number={3},
  pages={223--235},
  year={2023},
  publisher={Wiley Online Library}
}
```
