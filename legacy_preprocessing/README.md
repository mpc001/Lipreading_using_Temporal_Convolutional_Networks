### Pre-processing

* To get mouth ROIs

Run mouth cropping script to save grayscale mouth ROIs. We assume you save cropped mouths to *`$TCN_LIPREADING_ROOT/datasets/visual_data/`*. You can choose `--testset-only` to produce testing set.

```Shell
python crop_mouth_from_video.py --video-direc <LRW-DIREC> \
                                --landmark-direc <LANDMARK-DIREC> \
                                --save-direc <MOUTH-ROIS-DIRECTORY> \
                                --convert-gray \
                                --testset-only
```
