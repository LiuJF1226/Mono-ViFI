from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import numpy as np
import h5py
from PIL import Image
import torch
from torchvision import transforms
from .mono_dataset import MonoDataset

CROP = 16

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    # norm = np.array(h5f['norm'])
    # norm = np.transpose(norm, (1,2,0))
    # valid_mask = np.array(h5f['mask'])

    return rgb, depth

class NYUDataset(MonoDataset):

    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        if self.is_train:
            self.full_res_shape = (480-CROP*2, 640-CROP*2) 
        else:
            self.full_res_shape = (427, 561) 
            self.loader = h5_loader

        self.K = self._get_intrinsics()

    def __getitem__(self, index):
        # for testing
        if not self.is_train:
            line = self.filenames[index]
            line = os.path.join(self.data_path, line)
            rgb, depth = self.loader(line) 
            h, w, c = rgb.shape

            rgb = rgb[44: 471, 40: 601, :]
            depth = depth[44: 471, 40: 601]
            rgb = Image.fromarray(rgb)
            rgb = self.to_tensor(self.resize[0](rgb))
            depth = torch.tensor(depth)

            # K = self.K.copy()
            # K[0, :] *= self.width
            # K[1, :] *= self.height
            return rgb, depth

        inputs = {}

        do_color_aug = random.random() > 0.5
        do_flip = random.random() > 0.5

        line = self.filenames[index].split()
        frame_name = line[0]

        # if do_flip:
        #     if frame_name in ['nyu_depth_v2/bedroom_0086/r-1315251252.453265-3131793654.ppm', 'nyu_depth_v2/living_room_0011/r-1295837629.903855-1295712361.ppm']:
        #         do_flip = False
        #         print("-----------")
        #         print("meet not flip vps frame {}\n do_flip is False".format(frame_name))
        #         print("-----------")

        line = [os.path.join(self.data_path, l) for l in line]
        
        for ind, i in enumerate([0, -4, -3, -2, -1, 1, 2, 3, 4]):
            if not i in set(self.frame_idxs):
                continue
            inputs[("color", i, -1)] = self.get_color(line[ind], do_flip)


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in [0, -4, -3, -2, -1, 1, 2, 3, 4]:
            if not i in set(self.frame_idxs):
                continue

            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
    
        return inputs

    def get_color(self, fp, do_flip):
        color = self.loader(fp)
        color = self._undistort(np.array(color))

        if do_flip:
            color = cv2.flip(color, 1)

        h, w, c = color.shape
        color = color[CROP: h-CROP, CROP: w-CROP, :]

        return Image.fromarray(color)


    def _get_intrinsics(self):
        h, w = self.full_res_shape
        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        if self.is_train:
            cx = (3.2558244941119034e+02 - CROP) / w
            cy = (2.5373616633400465e+02 - CROP) / h
        else:
            cx = (3.2558244941119034e+02 - 40) / w
            cy = (2.5373616633400465e+02 - 44) / h
        intrinsics =np.array([[fx, 0., cx, 0.], 
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics

    def _undistort(self, image):
        k1 =  2.0796615318809061e-01
        k2 = -5.8613825163911781e-01
        p1 = 7.2231363135888329e-04
        p2 = 1.0479627195765181e-03
        k3 = 4.9856986684705107e-01

        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02

        kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([[k1, k2, p1, p2, k3]])
        image = cv2.undistort(image, kmat, dist)
        return image