# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from networks import *
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--pretrained_path',
                        type=str,
                        help="path of model checkpoint to load")
    parser.add_argument("--backbone",
                        type=str,
                        default="ResNet18",
                        choices=["ResNet18", "ResNet50", "LiteMono", "DHRNet"])
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--save_npy",
                        help='if set, saves numpy files of predicted disps/depths',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.pretrained_path is not None

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.pretrained_path)
    model = torch.load(args.pretrained_path, map_location='cpu')

    if args.backbone == "ResNet18":
        depth_encoder = monodepth2.DepthEncoder(18, 0)
        depth_decoder = monodepth2.DepthDecoder(depth_encoder.num_ch_enc, range(1))
    elif args.backbone == "ResNet50":
        depth_encoder = monodepth2.DepthEncoder(50, 0)      
        depth_decoder = monodepth2.DepthDecoder(depth_encoder.num_ch_enc, range(1))       
    elif args.backbone == "DHRNet":
        depth_encoder = DHRNet.DepthEncoder(18, 0)         
        depth_decoder = DHRNet.DepthDecoder(depth_encoder.num_ch_enc, range(1))          
    elif args.backbone == "LiteMono":
        depth_encoder = LiteMono.DepthEncoder(model='lite-mono', drop_path_rate=0.2,
                        width=args.width, height=args.height)
        depth_decoder = LiteMono.DepthDecoder(depth_encoder.num_ch_enc, range(1))

    depth_encoder.load_state_dict({k: v for k, v in model["encoder"].items() if k in depth_encoder.state_dict()})
    depth_decoder.load_state_dict({k: v for k, v in model["depth"].items() if k in depth_decoder.state_dict()})

    depth_encoder.cuda().eval()
    depth_decoder.cuda().eval()
    feed_height = args.height
    feed_width = args.width


    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = depth_encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp_resized, args.min_depth, args.max_depth)
            if args.save_npy:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            if args.save_npy:
                print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)