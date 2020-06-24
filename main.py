#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lipreading.utils import load_json, save2npz
from lipreading.model import Lipreading
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    # -- train
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    # -- test
    parser.add_argument('--model-path', type=str, default='./models/ckpt.best.pth.tar', help='Pretrained model pathname')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configiguration with json format')

    args = parser.parse_args()
    return args


args = load_args()


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def evaluate(model, dset_loader):
    model.eval()
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

    print('{} in total\tCR: {}'.format( len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))
    return


def get_model():
    if os.path.exists(args.config_path):
        args_loaded = load_json( args.config_path)
        args.backbone_type = args_loaded['backbone_type']
        args.width_mult = args_loaded['width_mult']
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                      }

    return Lipreading( num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       backbone_type=args.backbone_type,
                       width_mult=args.width_mult,
                       extract_feats=args.extract_feats).cuda()


def main():

    model = get_model()

    assert os.path.isfile(args.model_path), "File path does not exist. Path input: {}".format(args.model_path)
    model.load_state_dict( torch.load(args.model_path)["model_state_dict"], strict=True)

    if args.mouth_patch_path:
        save2npz( args.mouth_embedding_out_path, data = extract_feats(model).cpu().detach().numpy())
        return   
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    evaluate(model, dset_loaders['test'])

if __name__ == '__main__':
    main()
