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

STEREO_SCALE_FACTOR = 5.4

def eval_args():
    parser = argparse.ArgumentParser(description='Evaluation Parser')

    parser.add_argument('--pretrained_path',
                        type=str,
                        help="path of model checkpoint to load")
    parser.add_argument("--backbone",
                        type=str,
                        help="backbone of depth encoder",
                        default="ResNet18",
                        choices=["ResNet18", "ResNet50", "LiteMono", "DHRNet"])
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
    parser.add_argument("--use_stereo",
                        help="if set, uses stereo pair for training",
                        action="store_true")
    ## paths of test datasets
    parser.add_argument('--kitti_path',
                        type=str,
                        help="data path of KITTI, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--make3d_path',
                        type=str,
                        help="data path of Make3D, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--nyuv2_path',
                        type=str,
                        help="data path of NYU v2, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--cityscapes_path',
                        type=str,
                        help="data path of Cityscapes, do not set if you do not want to evaluate on this dataset")
    args = parser.parse_args()
    return args

def compute_errors(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


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

    depth_encoder.load_state_dict({k: v for k, v in model["encoder"].items() if k in depth_encoder.state_dict()})
    depth_decoder.load_state_dict({k: v for k, v in model["depth"].items() if k in depth_decoder.state_dict()})

    depth_encoder.cuda().eval()
    depth_decoder.cuda().eval()
    input_color = torch.ones(1, 3, args.height, args.width).cuda()
    flops, params, flops_e, params_e, flops_d, params_d = profile_once(depth_encoder, depth_decoder, input_color)
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d) + "\n")

    return depth_encoder, depth_decoder

def test_kitti(args, dataloader, depth_encoder, depth_decoder, eval_split='eigen'):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_path = os.path.join(os.path.dirname(__file__), "splits", "kitti", eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_disps = []
    for data in dataloader:
        input_color = data[("color", 0, 0)].cuda()
        if args.post_process:
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
        output = depth_decoder(depth_encoder(input_color))
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
        if args.use_stereo:
            pred_depth *= STEREO_SCALE_FACTOR
        else:
            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio  
        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_depth_errors(gt_depth, pred_depth))

    if not args.use_stereo:
        ratios = torch.tensor(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))

def test_make3d(args, dataloader, depth_encoder, depth_decoder):
    pred_depths = []
    gt_depths = []
    for data in dataloader:
        input_color = data["color"].cuda()
        if args.post_process:
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

        output = depth_decoder(depth_encoder(input_color))
        pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
        pred_disp = pred_disp[:, 0]

        if args.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], torch.flip(pred_disp[N:], [2]))
        
        gt_depth = data["depth"]
        _, h, w = gt_depth.shape
        pred_depth = 1 / pred_disp
        pred_depth = F.interpolate(pred_depth.unsqueeze(0), (h, w), mode="nearest")[0]
        pred_depths.append(pred_depth)
        gt_depths.append(gt_depth)
    pred_depths = torch.cat(pred_depths, dim=0)
    gt_depths = torch.cat(gt_depths, dim=0).cuda()

    errors = []
    ratios = []
    for i in range(pred_depths.shape[0]):    
        pred_depth = pred_depths[i]
        gt_depth = gt_depths[i]
        mask = (gt_depth > 0) & (gt_depth < 70)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if args.use_stereo:
            pred_depth *= STEREO_SCALE_FACTOR
        else:
            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio         
        pred_depth[pred_depth > 70] = 70
        errors.append(compute_errors(gt_depth, pred_depth))

    if not args.use_stereo:
        ratios = torch.tensor(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))

def test_nyuv2(args, dataloader, depth_encoder, depth_decoder):
    pred_depths = []
    gt_depths = []
    for (color, depth) in dataloader:
        input_color = color.cuda()
        if args.post_process:
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

        output = depth_decoder(depth_encoder(input_color))
        pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
        pred_disp = pred_disp[:, 0]

        if args.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], torch.flip(pred_disp[N:], [2]))
        
        gt_depth = depth
        _, h, w = gt_depth.shape
        pred_depth = 1 / pred_disp
        pred_depth = F.interpolate(pred_depth.unsqueeze(0), (h, w), mode="nearest")[0]
        pred_depths.append(pred_depth)
        gt_depths.append(gt_depth)
    pred_depths = torch.cat(pred_depths, dim=0)
    gt_depths = torch.cat(gt_depths, dim=0).cuda()

    errors = []
    ratios = []
    for i in range(pred_depths.shape[0]):    
        pred_depth = pred_depths[i]
        gt_depth = gt_depths[i]
        mask = (gt_depth > 0) & (gt_depth < 10)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        ratio = torch.median(gt_depth) / torch.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio         
        pred_depth[pred_depth > 10] = 10
        errors.append(compute_depth_errors(gt_depth, pred_depth))

    ratios = torch.tensor(ratios)
    med = torch.median(ratios)
    std = torch.std(ratios / med)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))


def test_cityscapes(args, dataloader, depth_encoder, depth_decoder):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_path = os.path.join(os.path.dirname(__file__), "splits", "cityscapes", "gt_depths")

    pred_disps = []
    for data in dataloader:
        input_color = data[("color", 0, 0)].cuda()
        if args.post_process:
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
        output = depth_decoder(depth_encoder(input_color))
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
       
        if args.use_stereo:
            pred_depth *= STEREO_SCALE_FACTOR
        else:
            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio  
        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_depth_errors(gt_depth, pred_depth))

    if not args.use_stereo:
        ratios = torch.tensor(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))  

def main(args):

    depth_encoder, depth_decoder = load_model(args)
    input_resolution = (args.height, args.width)
    
    print(" Evaluated at resolution {} * {}".format(input_resolution[0], input_resolution[1]))
    if args.post_process:
        print(" Post-process is used")
    else:
        print(" No post-process")
    if args.use_stereo:
        print(" Stereo evaluation - disabling median scaling")
        print(" Scaling by {} \n".format(STEREO_SCALE_FACTOR))
    else:
        print(" Mono evaluation - using median scaling \n")

    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    if args.kitti_path:
        # evaluate on eigen split
        print(" Evaluate on KITTI with eigen split:")
        filenames = readlines(os.path.join(splits_dir, "kitti", "eigen", "test_files.txt")) 
        dataset = datasets.KITTIRAWDataset(args.kitti_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_kitti(args, dataloader, depth_encoder, depth_decoder, "eigen")

        ## evaluate on eigen_benchmark split
        print(" Evaluate on KITTI with eigen_benchmark split (improved groundtruth):")
        filenames = readlines(os.path.join(splits_dir, "kitti", "eigen_benchmark", "test_files.txt")) 
        dataset = datasets.KITTIRAWDataset(args.kitti_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_kitti(args, dataloader, depth_encoder, depth_decoder, "eigen_benchmark")

    if args.make3d_path:
        print(" Evaluate on Make3D:")
        filenames = readlines(os.path.join(splits_dir, "make3d", "test_files.txt"))
        dataset = datasets.Make3DDataset(args.make3d_path, filenames, input_resolution)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_make3d(args, dataloader, depth_encoder, depth_decoder)

    if args.nyuv2_path:
        print(" Evaluate on NYU Depth v2:")
        filenames = readlines(os.path.join(splits_dir, "nyuv2", "test_files.txt"))
        dataset = datasets.NYUDataset(args.nyuv2_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_nyuv2(args, dataloader, depth_encoder, depth_decoder)

    if args.cityscapes_path:
        print(" Evaluate on Cisyscapes:")
        filenames = readlines(os.path.join(splits_dir, "cityscapes", "test_files.txt"))
        dataset = datasets.CityscapesDataset(args.cityscapes_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_cityscapes(args, dataloader, depth_encoder, depth_decoder)  

if __name__ == '__main__':
    args = eval_args()
    main(args)