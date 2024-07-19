import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets
from networks import *
from utils import *
from layers import disp_to_depth, compute_depth_errors
import warnings

warnings.filterwarnings("ignore")


def eval_args():
    parser = argparse.ArgumentParser(description='Evaluation Parser')

    parser.add_argument('--pretrained_path',
                        type=str,
                        help="path of model checkpoint to load")
    parser.add_argument('--training_data',
                        type=str,
                        default="kitti",
                        choices=["kitti", "cityscapes"])
    parser.add_argument("--backbone",
                        type=str,
                        help="backbone of depth encoder",
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
    parser.add_argument("--num_workers",
                        type=int,
                        help="number of dataloader workers",
                        default=12)
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)
    parser.add_argument("--post_process",
                        help="if set will perform the flipping post processing "
                            "from the original monodepth paper",
                        action="store_true")

    ## paths of test datasets
    parser.add_argument('--kitti_path',
                        type=str,
                        help="data path of KITTI, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--cityscapes_path',
                        type=str,
                        help="data path of Cityscapes, do not set if you do not want to evaluate on this dataset")
    args = parser.parse_args()
    return args

args = eval_args()


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    l_mask = torch.tensor(l_mask).cuda()
    r_mask = torch.tensor(r_mask.copy()).cuda()
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def load_model(args):
    print("-> Loading weights from {}".format(args.pretrained_path))

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

    fusion_module = FusionModule(args, depth_encoder.num_ch_enc)

    depth_encoder.load_state_dict({k: v for k, v in model["encoder_mf"].items() if k in depth_encoder.state_dict()})
    depth_decoder.load_state_dict({k: v for k, v in model["depth_mf"].items() if k in depth_decoder.state_dict()})
    fusion_module.load_state_dict({k: v for k, v in model["fusion_module"].items() if k in fusion_module.state_dict()})

    depth_encoder.cuda().eval()
    depth_decoder.cuda().eval()
    fusion_module.cuda().eval()

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

    img_n1 = torch.ones(1, 3, args.height, args.width).cuda()
    img0 = torch.ones(1, 3, args.height, args.width).cuda()
    img_p1 = torch.ones(1, 3, args.height, args.width).cuda()
    embt = torch.tensor(0.5).view(1, 1, 1, 1).float().cuda()
    embt = embt.repeat(1, 1, 1, 1)
    flow_0_n1, flow_0_p1, merge_mask_01 = model_vfi(img_n1, img_p1, embt, onlyFlow=True)  
    flops_vfi, params_vfi = profile(model_vfi, inputs=(img_n1, img_p1, embt, None, (1.0, 0.5), True,), verbose=False)   

    feats_n1 = depth_encoder(img_n1)
    feats_p1 = depth_encoder(img_p1)
    feats_0 = depth_encoder(img0)
    flops_e, params_e = profile(depth_encoder, inputs=(img_n1,), verbose=False)

    
    feats = [feats_n1, feats_0, feats_p1]
    flows = [flow_0_n1, flow_0_p1]
    flops_fm, params_fm = profile(fusion_module, inputs=(feats, flows, merge_mask_01,), verbose=False)
    feats = fusion_module(feats, flows, merge_mask_01)

    
    flops_d, params_d = profile(depth_decoder, inputs=(feats,), verbose=False)

    params = params_vfi + params_e + params_fm + params_d
    flops_single = flops_vfi + 3*flops_e + flops_fm + flops_d
    flops_average = flops_vfi + flops_e + flops_fm + flops_d
    flops_single, flops_average, params = clever_format([flops_single, flops_average, params], "%.3f")
    print("\n" + (" flops when processing a single frame: {0}, \n average flops per frame when processing a video: {1}, \n params: {2}").format(flops_single, flops_average, params) + "\n")

    return model_vfi, depth_encoder, depth_decoder, fusion_module


def test_kitti_mf(args, dataloader, model_vfi, depth_encoder, depth_decoder, fusion_module, eval_split='eigen'):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_path = os.path.join(os.path.dirname(__file__), "splits", "kitti", eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_disps = []
    for data in dataloader:
        img_n1 = data[("color", -1, 0)].cuda()
        img_p1 = data[("color", 1, 0)].cuda()
        img_0 = data[("color", 0, 0)].cuda()
        if args.post_process:
            img_n1 = torch.cat((img_n1, torch.flip(img_n1, [3])), 0)
            img_p1 = torch.cat((img_p1, torch.flip(img_p1, [3])), 0)
            img_0 = torch.cat((img_0, torch.flip(img_0, [3])), 0)
        embt = torch.tensor(0.5).view(1, 1, 1, 1).float().cuda()
        embt = embt.repeat(img_n1.shape[0], 1, 1, 1)
        flow_0_n1, flow_0_p1, merge_mask_01 = model_vfi(img_n1, img_p1, embt, onlyFlow=True)

        feats_n1 = depth_encoder(img_n1)
        feats_p1 = depth_encoder(img_p1)
        feats_0 = depth_encoder(img_0)

        feats = [feats_n1, feats_0, feats_p1]
        flows = [flow_0_n1, flow_0_p1]
        feats = fusion_module(feats, flows, merge_mask_01)
        output = depth_decoder(feats)  
        pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
        pred_disp = pred_disp[:, 0]
        if args.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], torch.flip(pred_disp[N:], [2]))
        pred_disps.append(pred_disp)
    pred_disps = torch.cat(pred_disps, dim=0)

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = torch.from_numpy(gt_depths[i]).cuda()
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i:i+1].unsqueeze(0)
        pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=True)
        pred_depth = 1 / pred_disp[0, 0, :]
        if eval_split == "eigen":
            mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            crop_mask = torch.zeros_like(mask)
            crop_mask[
                    int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            mask = mask * crop_mask
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = torch.median(gt_depth) / torch.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio  
        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_depth_errors(gt_depth, pred_depth))


    ratios = torch.tensor(ratios)
    med = torch.median(ratios)
    std = torch.std(ratios / med)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))


def test_cityscapes_mf(args, model_vfi, dataloader, depth_encoder, depth_decoder, fusion_module):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_path = os.path.join(os.path.dirname(__file__), "splits", "cityscapes", "gt_depths")

    pred_disps = []
    for data in dataloader:
        img_n1 = data[("color", -1, 0)].cuda()
        img_p1 = data[("color", 1, 0)].cuda()
        img_0 = data[("color", 0, 0)].cuda()
        if args.post_process:
            img_n1 = torch.cat((img_n1, torch.flip(img_n1, [3])), 0)
            img_p1 = torch.cat((img_p1, torch.flip(img_p1, [3])), 0)
            img_0 = torch.cat((img_0, torch.flip(img_0, [3])), 0)
        embt = torch.tensor(0.5).view(1, 1, 1, 1).float().cuda()
        embt = embt.repeat(img_n1.shape[0], 1, 1, 1)
        flow_0_n1, flow_0_p1, merge_mask_01 = model_vfi(img_n1, img_p1, embt, onlyFlow=True)

        feats_n1 = depth_encoder(img_n1)
        feats_p1 = depth_encoder(img_p1)
        feats_0 = depth_encoder(img_0)

        feats = [feats_n1, feats_0, feats_p1]
        flows = [flow_0_n1, flow_0_p1]
        feats = fusion_module(feats, flows, merge_mask_01)
        output = depth_decoder(feats)  
        pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
        pred_disp = pred_disp[:, 0]
        if args.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], torch.flip(pred_disp[N:], [2]))
        pred_disps.append(pred_disp)

    pred_disps = torch.cat(pred_disps, dim=0)
    
    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = np.load(os.path.join(gt_path, str(i).zfill(3) + '_depth.npy'))
        gt_height, gt_width = gt_depth.shape[:2]
        # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
        gt_height = int(round(gt_height * 0.75))
        gt_depth = torch.from_numpy(gt_depth[:gt_height]).cuda()
        pred_disp = pred_disps[i:i+1].unsqueeze(0)
        pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=True)
        pred_depth = 1 / pred_disp[0, 0, :]

        # when evaluating cityscapes, we centre crop to the middle 50% of the image.
        # Bottom 25% has already been removed - so crop the sides and the top here
        gt_depth = gt_depth[256:, 192:1856]
        pred_depth = pred_depth[256:, 192:1856]

        mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = torch.median(gt_depth) / torch.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio  
        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_depth_errors(gt_depth, pred_depth))

    ratios = torch.tensor(ratios)
    med = torch.median(ratios)
    std = torch.std(ratios / med)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))  

def main(args):

    model_vfi, depth_encoder, depth_decoder, fusion_module = load_model(args)
    input_resolution = (args.height, args.width)
    
    print(" Evaluated at resolution {} * {}".format(input_resolution[0], input_resolution[1]))
    if args.post_process:
        print(" Post-process is used")
    else:
        print(" No post-process")
    print(" Mono evaluation - using median scaling \n")

    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    if args.kitti_path:
        # evaluate on eigen split
        print(" Evaluate on KITTI with eigen split:")
        filenames = readlines(os.path.join(splits_dir, "kitti", "eigen", "test_files.txt")) 
        dataset = datasets.KITTIRAWDataset(args.kitti_path, filenames, input_resolution[0], input_resolution[1], [0,-1,1], 1, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_kitti_mf(args, dataloader, model_vfi, depth_encoder, depth_decoder, fusion_module, "eigen")

        ## evaluate on eigen_benchmark split
        print(" Evaluate on KITTI with eigen_benchmark split (improved groundtruth):")
        filenames = readlines(os.path.join(splits_dir, "kitti", "eigen_benchmark", "test_files.txt")) 
        dataset = datasets.KITTIRAWDataset(args.kitti_path, filenames, input_resolution[0], input_resolution[1], [0,-1,1], 1, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_kitti_mf(args, dataloader, model_vfi, depth_encoder, depth_decoder, fusion_module, "eigen_benchmark")

    if args.cityscapes_path:
        print(" Evaluate on Cisyscapes:")
        filenames = readlines(os.path.join(splits_dir, "cityscapes", "test_files.txt"))
        dataset = datasets.CityscapesDataset(args.cityscapes_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_cityscapes_mf(args, model_vfi, dataloader, depth_encoder, depth_decoder, fusion_module)  

if __name__ == '__main__':
    main(args)