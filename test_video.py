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
import copy
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
    parser.add_argument('--training_data',
                        type=str,
                        default="kitti",
                        choices=["kitti", "cityscapes"])
    parser.add_argument("--backbone",
                        type=str,
                        default="ResNet18",
                        choices=["ResNet18", "ResNet50", "LiteMono", "DHRNet"])
    parser.add_argument("--vfi_scale",
                        type=str,
                        help="the scale of IFRNet",
                        default="small",
                        choices=["large", "small"])
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size",
                        default=4)
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


def test_video(args):
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

 
    depth_decoder_mf = copy.deepcopy(depth_decoder)
    fusion_module = FusionModule(args, depth_encoder.num_ch_enc)
    depth_decoder_mf.load_state_dict({k: v for k, v in model["depth_mf"].items() if k in depth_decoder_mf.state_dict()})
    fusion_module.load_state_dict({k: v for k, v in model["fusion_module"].items() if k in fusion_module.state_dict()})

    depth_encoder.cuda().eval()
    depth_decoder.cuda().eval()
    depth_decoder_mf.cuda().eval()
    fusion_module.cuda().eval()

    feed_height = args.height
    feed_width = args.width

    model_vfi = IFRNet(scale=args.vfi_scale)
    if args.training_data == "kitti":
        if args.vfi_scale == "large":
            model_vfi.load_state_dict(torch.load("./weights/IFRNet_L_KITTI.pth")["VFI"])
        if args.vfi_scale == "small":
            model_vfi.load_state_dict(torch.load("./weights/IFRNet_S_KITTI.pth")["VFI"])
    if args.training_data == "cityscapes":
        if args.vfi_scale == "large":
            model_vfi.load_state_dict(torch.load("./weights/IFRNet_L_CS.pth")["VFI"])
        if args.vfi_scale == "small":
            model_vfi.load_state_dict(torch.load("./weights/IFRNet_S_CS.pth")["VFI"])
    model_vfi.cuda().eval()

    assert os.path.isdir(args.image_path)

    paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    paths = sorted(paths)
 
    output_directory = args.image_path

    print("-> Predicting on {:d} video frames".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        ims = []
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            img_ori = np.array(input_image).copy()
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0).cuda()

            if idx == 0 or idx == len(paths)-1:
                input_image_n1 = input_image
                input_image_p1 = input_image
            else:
                input_image_n1 = pil.open(paths[idx-1]).convert('RGB')
                input_image_n1 = input_image_n1.resize((feed_width, feed_height), pil.LANCZOS)
                input_image_n1 = transforms.ToTensor()(input_image_n1).unsqueeze(0).cuda()
                input_image_p1 = pil.open(paths[idx+1]).convert('RGB')
                input_image_p1 = input_image_p1.resize((feed_width, feed_height), pil.LANCZOS)
                input_image_p1 = transforms.ToTensor()(input_image_p1).unsqueeze(0).cuda()

            ### Single-frame prediction
            feats_0 = depth_encoder(input_image)
            outputs = depth_decoder(feats_0)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            if args.save_npy:
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                scaled_disp, depth = disp_to_depth(disp_resized, args.min_depth, args.max_depth)
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

            ### Multi-frame prediction
            embt = torch.tensor(0.5).view(1, 1, 1, 1).float().cuda()
            embt = embt.repeat(input_image.shape[0], 1, 1, 1)
            flow_0_n1, flow_0_p1, merge_mask_01 = model_vfi(input_image_n1, input_image_p1, embt, onlyFlow=True)

            feats_n1 = depth_encoder(input_image_n1)
            feats_p1 = depth_encoder(input_image_p1)
            feats = [feats_n1, feats_0, feats_p1]
            flows = [flow_0_n1, flow_0_p1]
            feats = fusion_module(feats, flows, merge_mask_01)
            outputs = depth_decoder_mf(feats)  
            disp_mf = outputs[("disp", 0)]
            disp_mf_resized = torch.nn.functional.interpolate(
                disp_mf, (original_height, original_width), mode="bilinear", align_corners=False)
            
            # Saving numpy file
            if args.save_npy:
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                scaled_disp_mf, depth_mf = disp_to_depth(disp_mf_resized, args.min_depth, args.max_depth)
                name_dest_npy = os.path.join(output_directory, "{}_mf_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp_mf.cpu().numpy())

            # Saving colormapped depth image
            disp_mf_resized_np = disp_mf_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_mf_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_mf_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im_mf = (mapper.to_rgba(disp_mf_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im_mf)
            name_dest_im = os.path.join(output_directory, "{}_mf_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            ### For the final gif output
            out = np.concatenate([img_ori, colormapped_im, colormapped_im_mf], axis=0)
            out = pil.fromarray(out)
            out = out.resize((feed_width//2, feed_height//2*3), pil.LANCZOS)
            ims.append(out)
        
        # name_dest = os.path.join(output_directory, "demo.gif")
        name_dest = "demo.gif"
        ims[0].save(name_dest, save_all=True, append_images=ims[1:], duration=150, loop=0)

        print(" Save output gif to:   {}".format(name_dest))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_video(args)