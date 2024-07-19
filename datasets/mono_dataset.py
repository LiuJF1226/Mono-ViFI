from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 use_affine=False,
                 resize_ratio=[1.2, 2.0],
                 rotate_range=[-5,5],
                 is_train=False,
                 doj_mask=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.resize_ratio_lower = resize_ratio[0]
        self.resize_ratio_upper = resize_ratio[1]

        self.rotate_angle_lower = rotate_range[0]
        self.rotate_angle_upper = rotate_range[1]

        self.use_affine = use_affine
        self.doj_mask = doj_mask

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        # self.load_depth = self.check_depth()
        self.load_depth = False
    
    def generate_mask(self, angle):
        white_image = Image.new('L', (self.width, self.height), 255)
        rotated_mask = white_image.rotate(angle, resample=Image.BILINEAR, expand=False)
        restore_mask = rotated_mask.rotate(-angle, resample=Image.BILINEAR, expand=False)
        return rotated_mask, restore_mask

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        if self.use_affine:
            resize_ratio = random.uniform(self.resize_ratio_lower, self.resize_ratio_upper)  
            height_re = int(self.height * resize_ratio)
            width_re = int(self.width * resize_ratio)        
            w0 = int((width_re - self.width) * random.random())
            h0 = int((height_re - self.height) * random.random())
            self.resize_local = transforms.Resize((height_re, width_re), interpolation=self.interp)
            box = (w0, h0, w0+self.width, h0+self.height)
            angle = random.uniform(self.rotate_angle_lower, self.rotate_angle_upper)      

            fs = 1 / resize_ratio
            R = torch.tensor([[np.cos(-np.pi/180*angle), np.sin(np.pi/180*angle), 0],
                [np.sin(-np.pi/180*angle), np.cos(-np.pi/180*angle), 0],
                [0,0,1]]).float()
            tmp = R @ torch.tensor([-fs*width_re/2, -fs*height_re/2, fs-1]) + torch.tensor([(width_re/2-w0)*fs, (height_re/2-h0)*fs, 0])
            Rc = inputs[("inv_K", 0)][:3,:3] @ R @ inputs[("K", 0)][:3,:3]
            tmp = inputs[("inv_K", 0)][:3,:3] @ tmp
            Rc_v = torch.zeros((3,3))
            Rc_v[:, 2] = tmp
            Rc = Rc + Rc_v
            inputs[("Rc")] = Rc
            inputs[("ratio_local")] = torch.tensor([resize_ratio])
            inputs[("angle")] = torch.tensor([angle])
            x0 = round(w0/resize_ratio)
            y0 = round(h0/resize_ratio)
            w = round(self.width/resize_ratio)
            h = round(self.height/resize_ratio)
            inputs[("box")] = torch.tensor([x0, y0, w, h])

            white_image = Image.new('L', (width_re, height_re), 255)
            rotated_img = white_image.rotate(angle, resample=Image.BILINEAR, expand=False)
            valid_mask_rec = rotated_img.crop(box)
            inputs[("valid_mask_rec")] = self.to_tensor(valid_mask_rec)
            inputs[("valid_mask_rec")][inputs[("valid_mask_rec")] > 0] = 1

            tmp = torch.nn.functional.interpolate(inputs[("valid_mask_rec")].unsqueeze(0), [h, w], mode="bilinear", align_corners=False)[0]
            new = torch.zeros((1, self.height, self.width))
            new[:, y0:y0+h, x0:x0+w] = tmp
            inputs[("valid_mask_cons")] = transforms.functional.rotate(new, angle=-angle, interpolation=2)
            inputs[("valid_mask_cons")][inputs[("valid_mask_cons")] > 0] = 1

        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    if self.use_affine:
                        if i == 0:
                            img = self.resize_local(inputs[(n, im, -1)])
                            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
                            inputs[(n + "_affine", im, i)] = img.crop(box)
                        else:
                            inputs[(n + "_affine", im, i)] = self.resize[i](inputs[(n + "_affine", im, i - 1)])                        

        for k in list(inputs):
            f = inputs[k]
            if "color" in k[0]:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        if type(self).__name__ in ["CityscapesDataset"] and self.doj_mask:
            if self.is_train:
                doj_mask = self.resize_local(inputs["doj_mask"])
                doj_mask = doj_mask.rotate(angle, resample=Image.BILINEAR, expand=False)
                inputs["doj_mask_affine"] = self.to_tensor(doj_mask.crop(box))
                doj_mask = self.resize_local(inputs["doj_mask-1"])
                doj_mask = doj_mask.rotate(angle, resample=Image.BILINEAR, expand=False)
                inputs["doj_mask-1_affine"] = self.to_tensor(doj_mask.crop(box))
                doj_mask = self.resize_local(inputs["doj_mask+1"])
                doj_mask = doj_mask.rotate(angle, resample=Image.BILINEAR, expand=False)
                inputs["doj_mask+1_affine"] = self.to_tensor(doj_mask.crop(box))
            inputs["doj_mask"] = self.to_tensor(self.resize[0](inputs["doj_mask"]))
            inputs["doj_mask-1"] = self.to_tensor(self.resize[0](inputs["doj_mask-1"]))
            inputs["doj_mask+1"] = self.to_tensor(self.resize[0](inputs["doj_mask+1"]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)

        if type(self).__name__ in ["CityscapesDataset"]:
            inputs.update(self.get_color(folder, frame_index, side, do_flip))       
            self.K = self.load_intrinsics(folder, frame_index)
            if self.doj_mask:
                inputs.update(self.get_doj_mask(folder, frame_index, side, do_flip))
        else:
            isValid = True
            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                    except:
                        ## for multi-frame testing, if a frame does not have previous or next frame
                        isValid = False

            if not isValid:
                inputs[("color", -1, -1)] = inputs[("color", 0, -1)].copy()
                inputs[("color", 1, -1)] = inputs[("color", 0, -1)].copy()

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(4):
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

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


