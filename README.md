# Lipreading using Temporal Convolutional Networks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lipreading-using-temporal-convolutional/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=lipreading-using-temporal-convolutional)

## Authors
[Pingchuan Ma](https://mpc001.github.io/), [Brais Martinez](http://braismartinez.org), [Stavros Petridis](https://ibug.doc.ic.ac.uk/people/spetridis), [Maja Pantic](https://ibug.doc.ic.ac.uk/people/mpantic).

## Content
[Deep Lipreading](#deep-lipreading)
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [How to install the environment](#how-to-install-environment)
- [How to prepare the dataset](#how-to-prepare-dataset)
- [How to test](#how-to-test)
- [How to extract embeddings](#how-to-extract-512-dim-embeddings)

[Model Zoo](#model-zoo)

[Citation](#citation)

[License](#license)

[Contact](#contact)



## Deep Lipreading
### Introduction

This is the respository of [Towards practical lipreading with distilled and efficient models](https://sites.google.com/view/audiovisual-speech-recognition#h.p_f7ihgs_dULaj) and [Lipreading using Temporal Convolutional Networks](https://sites.google.com/view/audiovisual-speech-recognition#h.p_jP6ptilqb75s). In this repository, we provide pre-trained models, network settings for end-to-end visual speech recognition (lipreading). We trained our model on [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). The network architecture is based on 3D convolution, ResNet-18 plus MS-TCN.

<div align="center"><img src="doc/pipeline.png" width="640"/></div>

By using this repository, you can achieve a performance of 87.9% on the LRW dataset. This reporsitory also provides a script for feature extraction.

### Preprocessing

As described in [our paper](https://arxiv.org/abs/2001.08702), each video sequence from the LRW dataset is processed by 1) doing face detection and face alignment, 2) aligning each frame to a reference mean face shape 3) cropping a fixed 96 × 96 pixels wide ROI from the aligned face image so that the mouth region is always roughly centered on the image crop 4) transform the cropped image to gray level.

You can run the pre-processing script provided in the [preprocessing](./preprocessing) folder to extract the mouth ROIs.

<table style="display: inline-table;">  
<tr><td><img src="doc/demo/original.gif", width="144"></td><td><img src="doc/demo/detected.gif" width="144"></td><td><img src="doc/demo/transformed.gif" width="144"></td><td><img src="doc/demo/cropped.gif" width="144"></td></tr>
<tr><td>0. Original</td> <td>1. Detection</td> <td>2. Transformation</td> <td>3. Mouth ROIs</td> </tr>
</table>

### How to install environment

1. Clone the repository into a directory. We refer to that directory as *`TCN_LIPREADING_ROOT`*.

```Shell
git clone --recursive https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks.git
```

2. Install all required packages.

```Shell
pip install -r requirements.txt
```

### How to prepare dataset

1. Download our pre-computed landmarks from [GoogleDrive](https://bit.ly/3huI1P5) or [BaiduDrive](https://bit.ly/2YIg8um) (key: kumy) and unzip them to *`$TCN_LIPREADING_ROOT/landmarks/`* folder.

2. Pre-process mouth ROIs using the script in the [preprocessing](./preprocessing) folder and save them to *`$TCN_LIPREADING_ROOT/datasets/`*.

3. Download a pre-trained model from [Model Zoo](#model-zoo) and put the model into the *`$TCN_LIPREADING_ROOT/models/`* folder.

### How to test

* To evaluate on LRW dataset:

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <DATA-DIRECTORY>
```

### How to extract 512-dim embeddings
We assume you have cropped the mouth patches and put them into *`<MOUTH-PATCH-PATH>`*. The mouth embeddings will be saved in the *`.npz`* format
* To extract 512-D feature embeddings from the top of ResNet-18:


```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --extract-feats \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --mouth-patch-path <MOUTH-PATCH-PATH> \
                                      --mouth-embedding-out-path <OUTPUT-PATH>
```

### Model Zoo
We plan to include more models in the future. We use a sequence of 29-frames with a size of 88 by 88 pixels to compute the FLOPs.

|       Architecture      |   Acc.   | FLOPs (G) | url | size (MB)|
|:-----------------------:|:--------:|:---------:|:---:|:----:|
|resnet18_mstcn_adamw_s3        |   87.9   |    10.31  |[GoogleDrive](https://bit.ly/3fo4w6P) or [BaiduDrive](https://bit.ly/2Zi5BaS) (key: bygn) |436.7|
|resnet18_mstcn                 |   85.5   |    10.31  |[GoogleDrive](https://bit.ly/2OiiQSw) or [BaiduDrive](https://bit.ly/3fhaq9X) (key: qwtm) |436.7|
|snv1x_tcn2x                    |   84.6   |    1.31   |[GoogleDrive](https://bit.ly/2Zl25wn) or [BaiduDrive](https://bit.ly/326dwtH) (key: f79d) |36.7|
|snv1x_dsmstcn3x                |   85.3   |    1.26   |[GoogleDrive](https://bit.ly/3ep9W06) or [BaiduDrive](https://bit.ly/3fo3RST) (key: 86s4) |37.5|
|snv1x_tcn1x                    |   82.7   |    1.12   |[GoogleDrive](https://bit.ly/38OHvri) or [BaiduDrive](https://bit.ly/32b213Z) (key: 3caa) |15.5|
|snv05x_tcn2x                   |   82.5   |    1.02   |[GoogleDrive](https://bit.ly/3iXLN4f) or [BaiduDrive](https://bit.ly/3h2WDED) (key: ej9e) |33.0|
|snv05x_tcn1x                   |   79.9   |    0.58   |[GoogleDrive](https://bit.ly/38LGQqL) or [BaiduDrive](https://bit.ly/2OgzsdB) (key: devg) |11.8|

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@article{ma2020towards,
  author       = "Ma, Pingchuan and Martinez, Brais and Petridis, Stavros and Pantic, Maja",
  title        = "Towards practical lipreading with distilled and efficient models",
  journal      = "arXiv preprint arXiv:2007.06504",
  year         = "2020",
}

@InProceedings{martinez2020lipreading,
  author       = "Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja",
  title        = "Lipreading using Temporal Convolutional Networks",
  booktitle    = "ICASSP",
  year         = "2020",
}

@InProceedings{petridis2018end,
  author       = "Petridis, Stavros and Stafylakis, Themos and Ma, Pingchuan and Cai, Feipeng and Tzimiropoulos, Georgios and Pantic, Maja",
  title        = "End-to-end audiovisual speech recognition",
  booktitle    = "ICASSP",
  year         = "2018",
}
```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[Pingchuan Ma](pingchuan.ma16[at]imperial.ac.uk)
```
