import glob
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class Camelyon16(Dataset):
    def __init__(self, root, file_path, transform=None):
        with open(file_path, "r") as f:
            data = f.read().split("\n")

        self.length = len(data) - 1
        self.transform = transform
        self.env = data
        self.root = root
        self.targets = [int(line.strip().split(",")[1]) for line in data[:-1]]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # print("train...,index=",index)
        assert index <= len(self), "index range error"
        img_path, label = self.env[index].strip().split(", ")
        img_path = os.path.join(self.root, img_path)
        # index += 1

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            # if img.layers == 1:
            #    print(img_path)

            img = self.transform(img)
            # print("image shape:".format(img.shape))

        return (img, int(label))


class PureTestDataset(Dataset):
    def __init__(self, root, transform=None):
        with open(root, "r") as f:
            data = f.read().split("\n")

        self.length = len(data) - 1
        self.transform = transform
        self.env = data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        # img_name = 30 whatever_x343_7343.png
        img_name = self.env[index].strip()
        folder = img_name.split("_")[0]
        # folder = folder + '_' + img_name.split('_')[1]
        # folder = 30 whatever
        folder = os.path.join(folder, "tissue")
        # folder = 30 whatever/tissue
        img_path = os.path.join(args.dataset_path, folder)
        # /mUSC_12_24_OUTPUT/30 Whatever/tissue
        img_path = os.path.join(img_path, img_name)
        # index += 1
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(img_path)
            return self[index + 1]

        if self.transform is not None:
            try:
                img = self.transform(img)
            except Exception as e:
                # print(e)
                # print(img_path)
                return self[index + 1]

        return (img, img_name)
