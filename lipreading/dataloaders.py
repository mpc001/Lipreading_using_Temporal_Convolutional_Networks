import os
import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, pad_packed_collate
from glob import glob

def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'raw_audio':

        preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('/mnt/c/Users/bohyh/Documents/GITHUB/SnLDetection/Lipreading_using_Temporal_Convolutional_Networks/data/babbleNoise_resample_16K.npy')),
                                    NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def get_data_loaders(args, modality = None, data_dir = None, batch_sampler = None):
    if modality is None:
        modality = args.modality
    if data_dir is None:
        data_dir = args.data_dir
    preprocessing = get_preprocessing_pipelines(modality)
    partition_list = [os.path.basename(f) for f in glob(os.path.join(data_dir, '**', '*'), recursive=True) if os.path.basename(f) in ['train', 'val', 'test']]
    tmp = []
    for partition in partition_list:
        if partition not in tmp:
            tmp.append(partition)
    partition_list = tmp
    # create dataset object for each partitio
    dsets = {partition: MyDataset(
                modality=modality,
                data_partition=partition,
                data_dir=data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz'
                ) for partition in partition_list}
    if batch_sampler is None:
        class_sample_count = {x: list(dsets[x].labels_count.values()) for x in partition_list}
        weights_per_label = {x: 1 / torch.Tensor(class_sample_count[x]) for x in partition_list}
        weights = {x: [weights_per_label[x][y[1]] for y in dsets[x].list.values()] for x in partition_list}
        sampler = {x: torch.utils.data.WeightedRandomSampler(weights[x], int(np.sum([*dsets[x].labels_count.values()])), replacement=True) for x in partition_list}
        batch_sampler = {x: list(torch.utils.data.BatchSampler(sampler[x], args.batch_size, True)) for x in partition_list}
    dset_loaders = {x: torch.utils.data.DataLoader(
                    dsets[x],
                    collate_fn=pad_packed_collate,
                    pin_memory=True,
                    num_workers=args.workers,
                    worker_init_fn=np.random.seed(1),
                    batch_sampler=batch_sampler[x]) for x in partition_list}
    return dset_loaders, batch_sampler
