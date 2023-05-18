# Lipreading using Temporal Convolutional Networks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/training-strategies-for-improved-lip-reading/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=training-strategies-for-improved-lip-reading)


## Authors
[Pingchuan Ma](https://mpc001.github.io/), [Brais Martinez](http://braismartinez.org), [Yujiang Wang](https://ibug.doc.ic.ac.uk/people/ywang), [Stavros Petridis](https://ibug.doc.ic.ac.uk/people/spetridis), [Jie Shen](https://ibug.doc.ic.ac.uk/people/jshen), [Maja Pantic](https://ibug.doc.ic.ac.uk/people/mpantic).


## Update

`2022-09-09`: We have released our DC-TCN models, see [here](#model-zoo).

`2021-06-09`: We have released our official training code, see [here](#how-to-train).

`2020-12-08`: We have released our audio-only models. see [here](#model-zoo).


## Content
[Deep Lipreading](#deep-lipreading)
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [How to install the environment](#how-to-install-environment)
- [How to prepare the dataset](#how-to-prepare-dataset)
- [How to train](#how-to-train)
- [How to test](#how-to-test)
- [How to extract embeddings](#how-to-extract-embeddings)

[Model Zoo](#model-zoo)

[Citation](#citation)

[License](#license)

[Contact](#contact)



## Deep Lipreading
### Introduction

This is the respository of [Training Strategies For Improved Lip-reading](https://sites.google.com/view/audiovisual-speech-recognition#h.p_f7ihgs_dULaj), [Towards Practical Lipreading with Distilled and Efficient Models](https://sites.google.com/view/audiovisual-speech-recognition#h.ob65v2y028mr) and [Lipreading using Temporal Convolutional Networks](https://sites.google.com/view/audiovisual-speech-recognition#h.p_jP6ptilqb75s). In this repository, we provide training code, pre-trained models, network settings for end-to-end visual speech recognition (lipreading). We trained our model on [LRW](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). The network architecture is based on 3D convolution, ResNet-18 plus MS-TCN.

<div align="center"><img src="doc/pipeline.png" width="640"/></div>

By using this repository, you can achieve a performance of 89.6% on the LRW dataset. This repository also provides a script for feature extraction.

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

1. Download a pre-trained model from [Model Zoo](#model-zoo) and put the model into the *`$TCN_LIPREADING_ROOT/models/`* folder.

2. For audio-only experiments, please pre-process audio waveforms using the script [extract_audio_from_video.py](./preprocessing/extract_audio_from_video.py) in the [preprocessing](./preprocessing) folder and save them to *`$TCN_LIPREADING_ROOT/datasets/audio_data/`*.

3. For VSR benchmarks reported in **Table 1**, please download our pre-computed landmarks from [GoogleDrive](https://bit.ly/3lEvDjs) or [BaiduDrive](https://bit.ly/3InhIYQ) (key: m00k) and unzip them to *`$TCN_LIPREADING_ROOT/landmarks/`* folder. please pre-process mouth ROIs using the script [crop_mouth_from_video.py](./preprocessing/crop_mouth_from_video.py) in the [preprocessing](./preprocessing) folder and save them to *`$TCN_LIPREADING_ROOT/datasets/visual_data/`*.

4. For VSR benchmarks reported in **Table 2**, please download our pre-computed landmarks from [GoogleDrive](https://bit.ly/3huI1P5) or [BaiduDrive](https://bit.ly/2YIg8um) (key: kumy) and unzip them to *`$TCN_LIPREADING_ROOT/landmarks/`* folder. please pre-process mouth ROIs using the script [crop_mouth_from_video.py](./legacy_preprocessing/crop_mouth_from_video.py) in the [legacy_preprocessing](./legacy_preprocessing) folder and save them to *`$TCN_LIPREADING_ROOT/datasets/visual_data/`*.


### How to train

1. Train a visual-only model.

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --config-path <MODEL-JSON-PATH> \
                                      --annonation-direc <ANNONATION-DIRECTORY> \
                                      --data-dir <MOUTH-ROIS-DIRECTORY>
```

2. Train an audio-only model.

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality audio \
                                      --config-path <MODEL-JSON-PATH> \
                                      --annonation-direc <ANNONATION-DIRECTORY> \
                                      --data-dir <AUDIO-WAVEFORMS-DIRECTORY>
```

We call the original LRW directory that includes timestamps (.txt) as *`<ANNONATION-DIRECTORY>`*.

3. Resume from last checkpoint.

You can pass the checkpoint path (`.pth` or `.pth.tar`) *`<CHECKPOINT-PATH>`* to the variable argument *`--model-path`*, and specify the *`--init-epoch`* to 1 to resume training.


### How to test

You need to specify *`<ANNONATION-DIRECTORY>`* if you use a model with utilising word boundaries indicators.

1. Evaluate the visual-only performance (lipreading).

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <MOUTH-ROIS-DIRECTORY> \
                                      --test
```

2. Evaluate the audio-only performance.

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality audio \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <AUDIO-WAVEFORMS-DIRECTORY>
                                      --test
```

### How to extract embeddings
We assume you have cropped the mouth patches and put them into *`<MOUTH-PATCH-PATH>`*. The mouth embeddings will be saved in the *`.npz`* format.
* To extract 512-D feature embeddings from the top of ResNet-18:


```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --extract-feats \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --mouth-patch-path <MOUTH-PATCH-PATH> \
                                      --mouth-embedding-out-path <OUTPUT-PATH>
```

### Model Zoo

<details open>

<summary>Table 1. Results of the audio-only and visual-only models on LRW. Mouth patches and audio waveforms are extracted in the <a href="https://github.com/mpc001/Lip-reading-with-Densely-Connected-Temporal-Convolutional-Networks/tree/main/preprocessing">preprocessing</a> folder.</summary>

<p> </p>

|           Architecture        |   Acc.   |                                            url                                           | size (MB)|
|:-----------------------------:|:--------:|:----------------------------------------------------------------------------------------:|:--------:|
|       **Audio-only**          |          |                                                                                          |          |
|resnet18_dctcn_audio_boundary  |   99.2   |[GoogleDrive](https://bit.ly/3KMQ6hh) or [BaiduDrive](https://bit.ly/3qbod9a) (key: w3jh) |    173   |
|resnet18_dctcn_audio           |   99.1   |[GoogleDrive](https://bit.ly/3KIVMsE) or [BaiduDrive](https://bit.ly/3RAwQ8V) (key: hw8e) |    173   |
|resnet18_mstcn_audio           |   98.9   |[GoogleDrive](https://bit.ly/3ReIfLK) or [BaiduDrive](https://bit.ly/3Qfuvz2) (key: bnhd) |    111   |
|      **Visual-only**          |          |                                                                                          |          |
|resnet18_dctcn_video_boundary  |   92.1   |[GoogleDrive](https://bit.ly/3KIzmYB) or [BaiduDrive](https://bit.ly/3cP40Tl) (key: jb7l) |    201   |
|resnet18_dctcn_video           |   89.6   |[GoogleDrive](https://bit.ly/3ejQqrB) or [BaiduDrive](https://bit.ly/3QhUwxA) (key: f3hd) |    201   |
|resnet18_mstcn_video           |   88.9   |[GoogleDrive](https://bit.ly/3AQTFOG) or [BaiduDrive](https://bit.ly/3THgNrF) (key: 0l63) |    139   |

</details>


<details open>

<summary>Table 2. Results of the visual-only models on LRW. Mouth patches are extracted in the <a href="https://github.com/mpc001/Lip-reading-with-Densely-Connected-Temporal-Convolutional-Networks/tree/main/legacy_preprocessing">legacy_preprocessing</a> folder.</summary>

<p> </p>

|           Architecture        |   Acc.   |                                            url                                           | size (MB)|
|:-----------------------------:|:--------:|:----------------------------------------------------------------------------------------:|:--------:|
|      **Visual-only**          |          |                                                                                          |          |
|snv1x_dsmstcn3x                |   85.3   |[GoogleDrive](https://bit.ly/3ep9W06) or [BaiduDrive](https://bit.ly/3fo3RST) (key: 86s4) |    36    |
|snv1x_tcn2x                    |   84.6   |[GoogleDrive](https://bit.ly/2Zl25wn) or [BaiduDrive](https://bit.ly/326dwtH) (key: f79d) |    35    |
|snv1x_tcn1x                    |   82.7   |[GoogleDrive](https://bit.ly/38OHvri) or [BaiduDrive](https://bit.ly/32b213Z) (key: 3caa) |    15    |
|snv05x_tcn2x                   |   82.5   |[GoogleDrive](https://bit.ly/3iXLN4f) or [BaiduDrive](https://bit.ly/3h2WDED) (key: ej9e) |    32    |
|snv05x_tcn1x                   |   79.9   |[GoogleDrive](https://bit.ly/38LGQqL) or [BaiduDrive](https://bit.ly/2OgzsdB) (key: devg) |    11    |

</details>


## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@INPROCEEDINGS{ma2022training,
    author={Ma, Pingchuan and Wang, Yujiang and Petridis, Stavros and Shen, Jie and Pantic, Maja},
    booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    title={Training Strategies for Improved Lip-Reading},
    year={2022},
    pages={8472-8476},
    doi={10.1109/ICASSP43922.2022.9746706}
}

@INPROCEEDINGS{ma2021lip,
  title={Lip-reading with densely connected temporal convolutional networks},
  author={Ma, Pingchuan and Wang, Yujiang and Shen, Jie and Petridis, Stavros and Pantic, Maja},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={2857-2866},
  year={2021},
  doi={10.1109/WACV48630.2021.00290}
}

@INPROCEEDINGS{ma2020towards,
  author={Ma, Pingchuan and Martinez, Brais and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Towards Practical Lipreading with Distilled and Efficient Models},
  year={2021},
  pages={7608-7612},
  doi={10.1109/ICASSP39728.2021.9415063}
}

@INPROCEEDINGS{martinez2020lipreading,
  author={Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Lipreading Using Temporal Convolutional Networks},
  year={2020},
  pages={6319-6323},
  doi={10.1109/ICASSP40776.2020.9053841}
}
```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[Pingchuan Ma](pingchuan.ma16[at]imperial.ac.uk)
```
