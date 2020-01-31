"""
Setup instructions

1. Download TinyImageNet
2. When asked for the "root" path, use the folder that contains 'train/' and 'val/'

Note that the validation set is used as a test set, as it is typically done with ImageNet (this is bad practice).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader, ImageFolder


class TinyImageNet(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(TinyImageNet, self).__init__(root + '/train', transform, target_transform)
        
    def __getitem__(self, item):
        sample, target = super(TinyImageNet, self).__getitem__(item)
        return sample, target


class TinyImageNetVal(Dataset):
    def __init__(self, root, class_to_idx, transform=None, target_transform=None):
        self.root = root + '/val'
        self.imgroot = self.root + '/images/'
        self.loader = pil_loader

        if class_to_idx is None:
            _, self.class_to_idx = TinyImageNet._find_classes(None, self.root)
        else:
            self.class_to_idx = class_to_idx
        self.samples = None
        self.make_dataset()

        self.transform = transform
        self.target_transform = target_transform
        
    def make_dataset(self):
        anno_file = Path(self.root) / 'val_annotations.txt'
        anno_data = anno_file.read_text().split('\n')
        headers = anno_data[0].split(',')
        anno_data = [l.strip().split(',') for l in anno_data[1:] if len(l.split(',')) >= 2]
        col_name = headers.index('name')
        col_class = headers.index('num')
        self.samples = [
            (row[col_name], self.class_to_idx[row[col_class]])
            for row in anno_data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        imgfile, target = self.samples[index]
        path = self.imgroot + imgfile
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)