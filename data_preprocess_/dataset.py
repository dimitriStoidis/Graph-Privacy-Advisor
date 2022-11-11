import os
import sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json
import random
import torchvision.transforms as transforms


full_transform = transforms.Compose([
    transforms.Resize((448, 448)),  #  transforms.Resize((448, 448)),
    transforms.ToTensor()])


class IPDataset_FromFolder(data.Dataset):
    def __init__(self, anno_dir, full_im_transform=None, dpath=''
            '/GIP/data_preprocess_', partition=''):
        super(IPDataset_FromFolder, self).__init__()

        self.anno_dir = anno_dir
        self.full_im_transform = full_im_transform

        f = open(str(dpath) + '/PrivacyAlert_' + str(partition) + '_private_files_path.txt', 'r')
        private_imgs = f.readlines()

        g = open(dpath + '/PrivacyAlert_' + str(partition) + '_public_files_path.txt', 'r')
        public_imgs = g.readlines()

        self.imgs = private_imgs + public_imgs
        # how to make continuous labels [0,1]
        self.labels = [0] * len(private_imgs) + [1] * len(public_imgs)

    def __getitem__(self, index):
        # For normalize
        if self.imgs[index].endswith('\n'):
            img = Image.open(self.imgs[index].split("\n")[0]).convert('RGB')
        else:
            img = Image.open(self.imgs[index]).convert('RGB')  # convert gray to rgb

        target = self.labels[index]

        if self.full_im_transform:
            full_im = self.full_im_transform(img)  # e.g; for index 10 full_im.shape = [3, 448, 448]
        else:
            full_im = img

        path = self.imgs[index].split('/')[-2:]  # e.g; ['train2017', '2017_80112549.jpg']
        path = os.path.join(self.anno_dir, path[1].split('.')[0] + '.json')

        image_name = path.split('/')[-1]  # e.g; '2017_80112549.json'
        (w, h) = img.size  # e.g; (1024, 1019)
        bboxes_objects = json.load(open(path))
        bboxes = torch.Tensor(bboxes_objects["bboxes"])

        max_rois_num = 12  # {detection threshold: max rois num} {0.3: 19, 0.4: 17, 0.5: 14, 0.6: 13, 0.7: 12}
        bboxes_14 = torch.zeros((max_rois_num, 4))

        if bboxes.size()[0] > max_rois_num:
            bboxes = bboxes[0:max_rois_num]

        if bboxes.size()[0] != 0:
            # re-scale, image size is wxh so change bounding boxes dimensions from wxh space to 448x448 range
            bboxes[:, 0::4] = 448. / w * bboxes[:, 0::4]
            bboxes[:, 1::4] = 448. / h * bboxes[:, 1::4]
            bboxes[:, 2::4] = 448. / w * bboxes[:, 2::4]
            bboxes[:, 3::4] = 448. / h * bboxes[:, 3::4]

            bboxes_14[0:bboxes.size(0), :] = bboxes

        categories = torch.IntTensor(max_rois_num + 1).fill_(-1)
        categories[0] = len(bboxes_objects['categories'])  # e.g; 5

        if categories[0] > max_rois_num:
            categories[0] = max_rois_num
        else:
            categories[0] = categories[0]
        end_idx = categories[0] + 1

        categories[1: end_idx] = torch.IntTensor(bboxes_objects['categories'])[
                                 0:categories[0]]  # e.g; [ 5,  0,  0,  0,  7, 72, -1, -1, -1, -1, -1, -1, -1]
        return target, full_im, categories, image_name

    def __len__(self):
        return len(self.imgs)
