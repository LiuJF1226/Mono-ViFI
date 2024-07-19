import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from imageio import imread


def random_crop(img0, img1, img2, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    img2 = img2[x:x+h, y:y+w, :]
    return img0, img1, img2

def random_reverse_channel(img0, img1, img2, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        img1 = img1[:, :, ::-1]
        img2 = img2[:, :, ::-1]
    return img0, img1, img2

def random_vertical_flip(img0, img1, img2, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        img1 = img1[::-1]
        img2 = img2[::-1]
    return img0, img1, img2

def random_horizontal_flip(img0, img1, img2, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        img1 = img1[:, ::-1]
        img2 = img2[:, ::-1]
    return img0, img1, img2

def random_reverse_time(img0, img1, img2, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img2
        img2 = img0
        img0 = tmp
    return img0, img1, img2


class KITTI_VFI_Dataset(Dataset):
    def __init__(self, data_path, filenames, height, width, is_train=False, img_ext='.png'):
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.img_ext = img_ext

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        line = self.filenames[index].split()
        folder = line[0]
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        if len(line) == 3:
            side = line[2]
        else:
            side = None

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        f_str = "{:010d}{}".format(frame_index-1, self.img_ext)
        img0 = imread(os.path.join(self.data_path, folder, "image_0{}/data".format(side_map[side]), f_str))
        img0 = cv2.resize(img0, (self.width, self.height))

        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        img1 = imread(os.path.join(self.data_path, folder, "image_0{}/data".format(side_map[side]), f_str))
        img1 = cv2.resize(img1, (self.width, self.height))

        f_str = "{:010d}{}".format(frame_index+1, self.img_ext)
        img2 = imread(os.path.join(self.data_path, folder, "image_0{}/data".format(side_map[side]), f_str)) 
        img2 = cv2.resize(img2, (self.width, self.height))       

        if self.is_train:
            # img0, img1, img2 = random_crop(img0, img1, img2, crop_size=(176, 608))
            img0, img1, img2 = random_crop(img0, img1, img2, crop_size=(160, 576))
            img0, img1, img2 = random_reverse_channel(img0, img1, img2, p=0.5)
            img0, img1, img2 = random_vertical_flip(img0, img1, img2, p=0.3)
            img0, img1, img2 = random_horizontal_flip(img0, img1, img2, p=0.5)
            img0, img1, img2 = random_reverse_time(img0, img1, img2, p=0.5)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, img1, img2, embt
    

class Cityscapes_VFI_Dataset(Dataset):
    def __init__(self, data_path, filenames, height, width, is_train=False, img_ext='.png'):
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.img_ext = img_ext


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        city, frame_name = self.filenames[index].split()
        color = imread(os.path.join(self.data_path, city, "{}.png".format(frame_name)))

        h = color.shape[0] // 3
        img0 = color[:h, :]
        img1 = color[h:2*h, :]
        img2 = color[2*h:, :]

        img0 = cv2.resize(img0, (self.width, self.height))
        img1 = cv2.resize(img1, (self.width, self.height))
        img2 = cv2.resize(img2, (self.width, self.height))       

        if self.is_train:
            img0, img1, img2 = random_crop(img0, img1, img2, crop_size=(176, 480))
            img0, img1, img2 = random_reverse_channel(img0, img1, img2, p=0.5)
            img0, img1, img2 = random_vertical_flip(img0, img1, img2, p=0.3)
            img0, img1, img2 = random_horizontal_flip(img0, img1, img2, p=0.5)
            img0, img1, img2 = random_reverse_time(img0, img1, img2, p=0.5)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, img1, img2, embt
