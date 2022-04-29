#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

from calendar import c
import pickle
from glob import glob
import os
from statistics import mode
import time
import random
import argparse
import optunity
import optunity.metrics
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import svm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    # parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'raw_audio', 'fusion'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    
    
    parser.add_argument('--SVM', default=False, action='store_true', help='fusion mode')
    parser.add_argument('--finetune', default=False, action='store_true', help='fusion mode')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')

    args = parser.parse_args()
    return args


args = load_args()

# torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True

class FusionNet(nn.Module):
    def __init__(self, input_size = 1536, n_classes = 3) -> None:
        super(FusionNet, self).__init__()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc4 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class Fusion():
    def __init__(self, args, models_path, is_SVM=True, logger=None) -> None:
        self.feat_models = []
        self.dset_loaders = []
        self.data_dirs = dict()
        self.sampler = None
        self.modalities = ['video', 'raw_audio']
        self.parent_dir= os.path.dirname(args.data_dir)
        self.full_finetune = False
        self.is_SVM = is_SVM
        self.epochs = args.epochs
        self.logger = logger

        self._load_fusion_model()

        if not is_SVM:
            self.init_optimizer = args.optimizer
            self._load_model_parameters()

        for modality in self.modalities:
            self._load_data_dirs(modality)
            self._load_dsets(args, modality)
            self._load_feats_models(modality, models_path[modality])

        for model in self.feat_models:
            model.eval()

    def _load_data_dirs(self, modality):
        feat = 'roi' if modality == 'video' else 'audio'
        self.data_dirs[modality] = os.path.join(*self.parent_dir.split('/'), feat)

    def _load_dsets(self, args, modality):
        ds_ldr, splr = get_data_loaders(args, modality, self.data_dirs[modality], self.sampler)
        self.dset_loaders.append(ds_ldr)
        if self.sampler is None:
            self.sampler = splr

    def _load_feats_models(self, modality, model_path):
        base_model = get_model_from_json(modality, fusion=True)
        self.feat_models.append(load_model(model_path, base_model, allow_size_mismatch=True).cuda())

    def _load_fusion_model(self):
        if self.is_SVM:
            self.fusion_model = None
        else:
            self.fusion_model = FusionNet().cuda()

    def _load_model_parameters(self):
        self.optimizer = self.init_optimizer
        self.lr = args.lr
        # -- get loss function
        self.criterion = nn.CrossEntropyLoss()
        # -- get optimizer
        self.optimizer = get_optimizer(self.optimizer, optim_policies=self.fusion_model.parameters(), lr=self.lr)
        # -- get learning rate scheduler
        self.scheduler = CosineScheduler(self.lr, self.epochs)

    def _train_SVM(self):
        self.pca = PCA(2)
        for batch_idx, (ds1, ds2) in enumerate(tqdm(zip(self.dset_loaders[0]['train'], self.dset_loaders[1]['train']))):
            input, lengths, labels, _ = ds1
            logits1 = self.feat_models[0](input.unsqueeze(1).cuda(), lengths=lengths)
            input, lengths, _, _ = ds2
            logits2 = self.feat_models[1](input.unsqueeze(1).cuda(), lengths=lengths)
            cat_logits = torch.cat((logits1, logits2), dim=-1)

            self.pca.fit(cat_logits.cpu().detach().numpy())
        
        for batch_idx, (ds1, ds2) in enumerate(tqdm(zip(self.dset_loaders[0]['train'], self.dset_loaders[1]['train']))):
            input, lengths, labels, _ = ds1
            logits1 = self.feat_models[0](input.unsqueeze(1).cuda(), lengths=lengths)
            input, lengths, _, _ = ds2
            logits2 = self.feat_models[1](input.unsqueeze(1).cuda(), lengths=lengths)
            cat_logits = torch.cat((logits1, logits2), dim=-1)

            pca_logits = self.pca.transform(cat_logits.cpu().detach().numpy())
            if batch_idx == 0:
                @optunity.cross_validated(x=pca_logits, y=labels, num_folds=10, num_iter=20)
                def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
                    clf = svm.SVC(kernel='linear', C=10 ** logC, gamma=10 ** logGamma, cache_size=1000).fit(x_train, y_train)
                    decision_values = clf.decision_function(x_test)
                    return optunity.metrics.accuracy(y_test, decision_values)
                
                hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
                print(hps)

                self.fusion_model = svm.SVC(kernel='linear', C = 10 ** hps['logC'], gamma = 10 ** hps['logGamma'], cache_size=1000, decision_function_shape='ovr')

            self.fusion_model.fit(pca_logits, labels)

    def _train_DNN(self):
        self.fusion_model.train()
        for batch_idx, (ds1, ds2) in enumerate(tqdm(zip(self.dset_loaders[0]['train'], self.dset_loaders[1]['train']))):
            input, lengths, labels, _ = ds1
            logits1 = self.feat_models[0](input.unsqueeze(1).cuda(), lengths=lengths)
            input, lengths, _, _ = ds2
            logits2 = self.feat_models[1](input.unsqueeze(1).cuda(), lengths=lengths)
            cat_logits = torch.cat((logits1, logits2), dim=-1)

            input, labels_a, labels_b, lam = mixup_data(cat_logits, F.one_hot(labels, num_classes=3).float().cuda(), args.alpha)
            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

            self.optimizer.zero_grad()
            logits = self.fusion_model(cat_logits.unsqueeze(1).cuda())

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(self.criterion, logits.squeeze(1))

            loss.backward()
            self.optimizer.step()

    def train_full_finetune(self, epoch):
        self.logits = torch.Tensor()
        if epoch == 0:
            for model in self.feat_models:
                model.train()

        self._train_DNN()

    def train(self):
        self.logits = torch.Tensor()
        if self.is_SVM:
            self._train_SVM()
        else:
            self._train_DNN()
            

    def evaluate(self, partition):
        intens_lst = ['0low', '0medium', '0high', '1subtle', '1low', '1medium', '1high', '2none']
        intensity_count = {x: [] for x in intens_lst}
        running_corrects = 0.
        if not self.is_SVM:
            self.fusion_model.eval()
            running_loss = 0.
        
        with torch.no_grad():
            self.true_labels = torch.Tensor().cuda()
            self.predictions = torch.Tensor().cuda()
            for batch_idx, (ds1, ds2) in enumerate(zip(self.dset_loaders[0][partition], self.dset_loaders[1][partition])):
                input, lengths, labels, _ = ds1
                logits1 = self.feat_models[0](input.unsqueeze(1).cuda(), lengths=lengths)
                input, lengths, labels, intensities = ds2
                logits2 = self.feat_models[1](input.unsqueeze(1).cuda(), lengths=lengths)
                cat_logits = torch.cat((logits1, logits2), dim=-1)

                if self.is_SVM:
                    pca_logits = self.pca.transform(cat_logits.cpu().numpy())
                    preds = torch.Tensor(self.fusion_model.predict(pca_logits)).cuda()
                    running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()
                else:
                    logits = self.fusion_model(cat_logits.cuda())
                    _, preds = torch.max(F.softmax(logits, dim=1), dim=1)
                    running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()
                    loss = self.criterion(logits.squeeze(1), F.one_hot(labels, num_classes=3).float().cuda())
                    running_loss += loss.item() * input.size(0)

                for i in range(len(intensities)):
                    try:
                        intensity_count[f'{labels[i]}{intensities[i]}'].append(int(preds.cpu()[i]))
                    except:
                        print(intensities, labels, preds.cpu())
                self.true_labels = torch.cat((self.true_labels, labels.cuda()), dim=-1).type(torch.int8)
                self.predictions = torch.cat((self.predictions, preds)).type(torch.int8)
            if self.logger is not None:
                self.logger.info("{} Confusion Matrix: \n{}".format(args.modality, confusion_matrix(self.true_labels.cpu(), self.predictions.cpu(), labels=[0, 1, 2])))

                asym_conf_matrix = np.zeros((8, 3), dtype=np.int32)
                for k, c in intensity_count.items():
                    for idx in c:
                        asym_conf_matrix[intens_lst.index(k)][idx] += 1

                self.logger.info(f"{args.modality} Intensity distribution: \n")
                self.logger.info(f"Preds:\t{'Laughs', 'Smiles', 'None'}")
                for i, k in enumerate(intensity_count.keys()):
                    self.logger.info(f"{k} ({np.sum(asym_conf_matrix[i])} files): \t{asym_conf_matrix[i]*100/np.sum(asym_conf_matrix[i])}")


            print('{} in total\tCR: {}'.format( len(self.dset_loaders[0][partition].dataset), running_corrects/len(self.dset_loaders[0][partition].dataset)))
        if self.is_SVM:
            return running_corrects/len(self.dset_loaders[0][partition].dataset)
        else:
            return running_corrects/len(self.dset_loaders[0][partition].dataset), running_loss/len(self.dset_loaders[0][partition].dataset)

    def load(self, fusion_model_path):
        if not self.is_SVM and fusion_model_path.endswith('.tar'):
            self.fusion_model = load_model(fusion_model_path, self.fusion_model)
        elif self.is_SVM and fusion_model_path.endswith('.sav'):
            self.fusion_model = pickle.load(open('./DATABASES/ndc/segmented/SVM_model.sav', 'rb'))
            self.pca = pickle.load(open('./DATABASES/ndc/segmented/PCA_model.sav', 'rb'))


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def evaluate(model, dset_loader, criterion, logger=None):
    intens_lst = ['0low', '0medium', '0high', '1subtle', '1low', '1medium', '1high', '2none']
    label_names = ['Laughs low', 'Laughs medium', 'Laughs high', 'Smiles subtle', 'Smiles low', 'Smiles medium', 'Smiles high', 'None']
    intensity_count = {x: [] for x in intens_lst}
    
    model.eval()

    running_loss = 0.
    running_corrects = 0.
    all_logits = np.zeros((len(dset_loader)*32, 768))
    all_alpha = np.zeros(len(dset_loader)*32)
    with torch.no_grad():
        true_labels = []
        predictions = []
        for batch_idx, (input, lengths, labels, intensities) in enumerate(tqdm(dset_loader)):
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)



        #     all_logits[batch_idx*32:(batch_idx+1)*32] = logits.cpu()
        #     tmp_alpha = [intens_lst.index(f'{x}{y}') for x, y in zip(labels, intensities)]
        #     all_alpha[batch_idx*32:(batch_idx+1)*32] = tmp_alpha

        # # PLOT TSNE
        # tsne_ = TSNE().fit_transform(all_logits)
        # colors = mcolors.CSS4_COLORS
        # cmap = [colors['gold'], colors['orange'], colors['red'], colors['aqua'], colors['deepskyblue'], colors['blue'], colors['indigo'], colors['gray']]
        
        # values_by_color = {x: [[], []] for x in cmap}
        # col = []
        # for i, v in enumerate(tsne_):
        #     values_by_color[cmap[int(all_alpha[i])]][0].append(v[0])
        #     values_by_color[cmap[int(all_alpha[i])]][1].append(v[1])
        # fig, ax = plt.subplots()
        # for k, v in values_by_color.items():
        #     ax.scatter(v[0], v[1], c=k, label=label_names[cmap.index(k)])

        # ax.legend()
        # plt.show()

            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)
            true_labels.extend(labels)
            predictions.extend(preds.cpu())
        
            for i in range(len(intensities)):
                try:
                    intensity_count[f'{labels[i]}{intensities[i]}'].append(int(preds.cpu()[i]))
                except:
                    print(intensities, labels, preds.cpu())
        if logger is not None:
            logger.info("{} Confusion Matrix: \n{}".format(args.modality, confusion_matrix(true_labels, predictions, labels=[0, 1, 2])))

            asym_conf_matrix = np.zeros((8, 3), dtype=np.int32)
            for k, c in intensity_count.items():
                for idx in c:
                    asym_conf_matrix[intens_lst.index(k)][idx] += 1

            logger.info(f"{args.modality} Intensity distribution: \n")
            logger.info(f"Preds:\t{'Laughs', 'Smiles', 'None'}")
            for i, k in enumerate(intensity_count.keys()):
                logger.info(f"{k} ({np.sum(asym_conf_matrix[i])} files): \t{asym_conf_matrix[i]*100/np.sum(asym_conf_matrix[i])}")


    print('{} in total\tCR: {}'.format( len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset)


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current learning rate: {}'.format(lr))

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.
    end = time.time()
    for batch_idx, (input, lengths, labels, _) in enumerate(dset_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()
        
        logits = model(input.unsqueeze(1).cuda(), lengths=lengths) # output shape = [batch_size, n_classes]

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item()*input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader)-1):
            update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

    return model


def get_model_from_json(modality = None, fusion=False):
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        "'.json' config path does not exist. Path input: {}".format(args.config_path)
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }
    if modality is None:
        modality = args.modality

    model = Lipreading( modality=modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        extract_feats=args.extract_feats,
                        fusion=fusion).cuda()
    calculateNorm2(model)
    return model


def fusion(args, logger, ckpt_saver, models_path, is_SVM = False, full_finetune=True):

    fusion_obj = Fusion(args, models_path, is_SVM, logger)
    # -- get models
    
    epoch = args.init_epoch
    
    if args.test:
        parti = 'test' if 'test' in list(fusion_obj.sampler.keys()) else 'train'
        if is_SVM:
            fusion_model_path = f"{os.path.join(*args.data_dir.split('/')[:-1])}/SVM_model.sav"
        elif args.model_path.endswith('.tar'):
            fusion_model_path = args.model_path
        else:
            fusion_model_path = args.model_path+"/ckpt.best.pth.tar"
        fusion_obj.load(fusion_model_path)
        logger.info('Model has been successfully loaded from {}'.format(fusion_model_path))
        fusion_obj.evaluate(parti)
        return

    while epoch < args.epochs:
        logger.info('-' * 10)
        if is_SVM:
            logger.info('Training PCA transformer and SVM-fusion model')
            fusion_obj.train()
            acc_avg_val = fusion_obj.evaluate('val')
            pickle.dump(fusion_obj.fusion_model, open(os.path.dirname(args.data_dir)+'/SVM_model.sav', 'wb'))
            pickle.dump(fusion_obj.pca, open(os.path.dirname(args.data_dir)+'/PCA_model.sav', 'wb'))
            break
        else:
            if epoch == args.init_epoch:
                logger.info('Training DNN fusion model')
            logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
            fusion_obj.train()
            acc_avg_val, loss_avg_val = fusion_obj.evaluate('val')
            logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val, showLR(fusion_obj.optimizer)))
            # -- save checkpoint
            save_dict = {
                'epoch_idx': epoch + 1,
                'model_state_dict': fusion_obj.fusion_model.state_dict(),
                'optimizer_state_dict': fusion_obj.optimizer.state_dict()
            }
            ckpt_saver.save(save_dict, acc_avg_val)
            fusion_obj.scheduler.adjust_lr(fusion_obj.optimizer, epoch)
        epoch += 1


    if is_SVM:
        fusion_obj.evaluate('test')
    else:
        if full_finetune:
            fusion_obj._load_model_parameters()
            f_epoch = 0
            f_epochs = 20
            while f_epoch < f_epochs:
                logger.info('Epoch {}/{}'.format(f_epoch, f_epochs - 1))
                fusion_obj.train_full_finetune(f_epoch)
                acc_avg_val, loss_avg_val = fusion_obj.evaluate('val')
                logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', f_epoch, loss_avg_val, acc_avg_val, showLR(fusion_obj.optimizer)))
                # -- save checkpoint
                save_dict = {
                    'epoch_idx': f_epoch + 1,
                    'model_state_dict': fusion_obj.fusion_model.state_dict(),
                    'optimizer_state_dict': fusion_obj.optimizer.state_dict()
                }
                ckpt_saver.save(save_dict, acc_avg_val)
                fusion_obj.scheduler.adjust_lr(fusion_obj.optimizer, f_epoch)
                f_epoch += 1
        best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
        _ = load_model(best_fp, fusion_obj.fusion_model)
        acc_avg_test, loss_avg_test = fusion_obj.evaluate('test')
        logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))


def main():

    # models_path = {'video': 'train_logs/tcn/fullndc_video_model_realfinetuned/ckpt.best.pth.tar', 'raw_audio': 'train_logs/tcn/fullndc_audio_model_realfinetuned/ckpt.best.pth.tar'}
    # models_path = {'video': 'train_logs/tcn/fullndc_video_model_scratch/ckpt.best.pth.tar', 'raw_audio': 'train_logs/tcn/fullndc_audio_model_scratch/ckpt.best.pth.tar'}
    models_path = {'video': 'train_logs/tcn/fullndc_video_model_finetuned/ckpt.best.pth.tar', 'raw_audio': 'train_logs/tcn/fullndc_audio_model_finetuned/ckpt.best.pth.tar'}
    # -- logging
    save_path = get_save_folder( args)
    print("Model and log being saved in: {}".format(save_path))
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    if args.modality == 'fusion':
        fusion(args, logger, ckpt_saver, models_path, args.SVM)
        return
        
    # -- get model
    model = get_model_from_json()
    # -- get dataset iterators
    dset_loaders, _ = get_data_loaders(args)
    
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args.optimizer, optim_policies=model.parameters(), lr=args.lr)
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)

    if args.model_path:
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info('Model and states have been successfully loaded from {}'.format( args.model_path ))
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            # for param in model.parameters():
            if args.finetune:
                for param in model.trunk.parameters():
                    param.requires_grad = False
            logger.info('Model has been successfully loaded from {}'.format( args.model_path ))
        # feature extraction
        if args.mouth_patch_path:
            save2npz( args.mouth_embedding_out_path, data = extract_feats(model).cpu().detach().numpy())
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion, logger=logger)
            logger.info('Test-time performance on partition {}: Loss: {:.4f}\tAcc:{:.4f}'.format( 'test', loss_avg_test, acc_avg_test))
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch-1)

    epoch = args.init_epoch
    while epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'], criterion, logger=logger)
        logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val, showLR(optimizer)))
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion, logger=logger)
    logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))

if __name__ == '__main__':
    main()
