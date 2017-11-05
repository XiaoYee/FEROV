#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
import cv2
import cPickle


class listDataset(Dataset):

    def __init__(self, root, shape=None, length = None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=1, num_workers=4):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = length
        self.height = shape[0]
        self.width  = shape[1]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train:
            # print imgpath
            img, label = self.load_data_label(imgpath)
            img   = torch.from_numpy(img).float()
            label = torch.from_numpy(label)

        else:
            img, label = self.load_data_label(imgpath)
            # print img
            img   = torch.from_numpy(img).float()
            label = torch.from_numpy(label)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)


    def load_data_label(self,imgpath):

        label = np.zeros(self.batch_size*self.length, dtype=np.uint8)
        classes = self.get_classes()
        file = open('info.pkl','rb')
        info = cPickle.load(file)
        dims = (self.height, self.width)
        seq = np.zeros((self.height, self.width, 3, self.batch_size*self.length), dtype=np.float32)

        # mean_file = 'ilsvrc_2012_mean.mat'
        # d = sio.loadmat(mean_file)
        # image_mean = d['mean_data'][:self.height, :self.width, ]

        video_length = info[imgpath]

        if video_length >= self.length:
            select_frame = sorted(random.sample(range(video_length), self.length))
            for m in range(self.length):
                img_file = os.path.join(imgpath, 'images{0:03d}.jpg'.format(select_frame[m]+1))
                img = cv2.imread(img_file)
                img = cv2.resize(img, dims[1::-1])
                # print img.shape
                choose = random.randint(0, 1)
                img = self.data_augmentation(img, img[:,::-1,:], choose)
                # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img_gray = img_gray/255.
                seq[:, :, :, m] = img
                label[m] = classes.index(imgpath.split('/')[-2])
        else:
            for k in range(self.length):
                if k+1 <= video_length:
                    img_file = os.path.join(imgpath, 'images{0:03d}.jpg'.format(k+1))
                else:
                    img_file = os.path.join(imgpath, 'images{0:03d}.jpg'.format(video_length))
                img = cv2.imread(img_file)
                img = cv2.resize(img, dims[1::-1])
                choose = random.randint(0, 1)
                img = self.data_augmentation(img, img[:,::-1,:], choose)
                # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img_gray = img_gray/255.
                seq[:, :, :, k] = img
                label[k] = classes.index(imgpath.split('/')[-2])
        # seq = seq[...] - np.tile(image_mean[...,np.newaxis], (1, 1, 1, seq.shape[3]))
        # data = np.transpose(seq, (2,0,1))
        data = np.transpose(seq, (3,2,0,1))

        return data, label

    def data_augmentation(self, img, img_flip, choose):
        if choose == 0:
            img_aug = img
        else:
            img_aug = img_flip
        return img_aug

        
    def get_classes(self):
        classes = []
        for line in open('TrainTestlist/classInd.txt'):
            classes.append(line.strip().split()[1])
        return classes

