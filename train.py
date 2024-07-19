import numpy as np
import time
import logging

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from options import opts
import torch.distributed as dist
import os
import json
from utils import *
from kitti_utils import *
from layers import *
import datasets
from networks import *
import copy
import shutil
import warnings

warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.exp_name)
        if self.opt.global_rank == 0:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.save_opts()
            if not os.path.exists(os.path.join(self.log_path, 'ckpt.pth')):
                setup_logging(os.path.join(self.log_path, 'logger.log'), rank=self.opt.global_rank)
                logging.info("Experiment is named: %s", self.opt.exp_name)
                logging.info("Saving to: %s", os.path.abspath(self.log_path))
                logging.info("GPU numbers: %d", self.opt.world_size)
                logging.info("Training dataset: %s", self.opt.dataset)
            else:
                setup_logging(os.path.join(self.log_path, 'logger.log'), filemode='a', rank=self.opt.global_rank)

            self.writers = {}
            for mode in ["train"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, "tensorboard", mode))
        if self.opt.world_size > 1:
            dist.barrier()

        self.device = torch.device('cuda', self.opt.local_rank)

        if self.opt.seed > 0:
            self.set_seed(self.opt.seed)
        else:
            cudnn.benchmark = True

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        self.ep_start = 0
        self.batch_start = 0
        self.step = 0

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes": datasets.CityscapesDataset,
                         "nyuv2": datasets.NYUDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset == "kitti":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.split, "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.eval_split, "{}_files.txt")
        elif self.opt.dataset == "kitti_odom":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", "odom", "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/kitti", "odom", "{}_files_09.txt")
        elif self.opt.dataset == "nyuv2":
            fpath = os.path.join(os.path.dirname(__file__), "splits/nyuv2", "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/nyuv2", "{}_files.txt")
        elif self.opt.dataset == "cityscapes":
            fpath = os.path.join(os.path.dirname(__file__), "splits/cityscapes", "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/cityscapes", "{}_files.txt")            
        else:
            pass

        train_filenames = readlines(fpath.format("train"))
        test_filenames = readlines(fpath_test.format("test"))
        img_ext = '.jpg' if self.opt.jpg else '.png'

        num_train_samples = len(train_filenames)
        self.num_steps_per_epoch = num_train_samples // self.opt.world_size // self.opt.batch_size
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs

        if self.opt.dataset == "cityscapes":
            train_dataset = self.dataset(
                self.opt.data_path_pre, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, self.opt.num_scales, self.opt.use_affine, is_train=True, img_ext=img_ext)
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, self.opt.num_scales, self.opt.use_affine, is_train=True, img_ext=img_ext)
        if self.opt.world_size > 1:
            self.sampler = datasets.CustomDistributedSampler(train_dataset, self.opt.seed)
        else:
            self.sampler = datasets.CustomSampler(train_dataset, self.opt.seed)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False, sampler=self.sampler, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        # for testing the model at the end of each epoch
        test_dataset = self.dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            [0,-1,1], self.opt.num_scales, is_train=False, img_ext=img_ext)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        if self.opt.dataset == "kitti":
            gt_path = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.eval_split, "gt_depths.npz")
            self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        elif self.opt.dataset == "cityscapes":
            gt_path = os.path.join(os.path.dirname(__file__), "splits", "cityscapes", "gt_depths")
            self.gt_depths = []
            for i in range(len(test_dataset)):
                gt_depth = np.load(os.path.join(gt_path, str(i).zfill(3) + '_depth.npy'))
                self.gt_depths.append(gt_depth)
        else:
            pass

        # create models
        if self.opt.backbone == "ResNet18":
            self.models["encoder"] = monodepth2.DepthEncoder(
                18, self.opt.weights_init == "pretrained")
            self.models["depth"] = monodepth2.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales))
        elif self.opt.backbone == "ResNet50":
            self.models["encoder"] = monodepth2.DepthEncoder(
                50, self.opt.weights_init == "pretrained")      
            self.models["depth"] = monodepth2.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales))    
        elif self.opt.backbone == "DHRNet":
            self.models["encoder"] = DHRNet.DepthEncoder(
                18, self.opt.weights_init == "pretrained")      
            self.models["depth"] = DHRNet.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales))                 
        elif self.opt.backbone == "LiteMono":
            self.models["encoder"] = LiteMono.DepthEncoder(model='lite-mono',
                                            drop_path_rate=0.2,
                                            width=self.opt.width, height=self.opt.height)
            model_dict = self.models["encoder"].state_dict()
            pretrained_dict = torch.load("./weights/lite-mono-pretrain.pth")['model']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
            model_dict.update(pretrained_dict)
            self.models["encoder"].load_state_dict(model_dict)
            self.models["depth"] = LiteMono.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales))

        ## create multi-frame depth model
        if self.opt.fuse_model_type == "shared_all":
            self.models["encoder_mf"] = self.models["encoder"]
            self.models["depth_mf"] = self.models["depth"]
        elif self.opt.fuse_model_type == "shared_encoder":
            self.models["encoder_mf"] = self.models["encoder"]
            self.models["depth_mf"] = copy.deepcopy(self.models["depth"])
        elif self.opt.fuse_model_type == "separate_all":
            self.models["encoder_mf"] = copy.deepcopy(self.models["encoder"])
            self.models["depth_mf"] = copy.deepcopy(self.models["depth"])
        self.models["fusion_module"] = FusionModule(self.opt, self.models["encoder_mf"].num_ch_enc)

        if self.use_pose_net:
            self.models["pose_encoder"] = posenet.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose"] = posenet.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        if self.opt.pretrained_path:
            if not self.opt.resume:
                self.load_pretrained_model()
            elif not os.path.exists(os.path.join(self.log_path, 'ckpt.pth')):
                self.load_pretrained_model()

        for k in self.models.keys():
            self.models[k].to(self.device)
            self.parameters_to_train += list(self.models[k].parameters())
            
        if self.opt.resume:
            checkpoint = self.load_ckpt()
     
        if self.opt.world_size > 1:
            for k in self.models.keys():
                self.models[k] = nn.SyncBatchNorm.convert_sync_batchnorm(self.models[k])
                self.models[k] = nn.parallel.DistributedDataParallel(self.models[k], device_ids=[self.opt.local_rank], output_device=self.opt.local_rank, find_unused_parameters=True)

        self.model_vfi_train = IFRNet(scale="large")
        self.model_vfi_test = IFRNet(scale="small")
        if self.opt.dataset == "kitti":
            self.model_vfi_train.load_state_dict(torch.load("./weights/IFRNet_L_KITTI.pth")["VFI"])
            self.model_vfi_test.load_state_dict(torch.load("./weights/IFRNet_S_KITTI.pth")["VFI"])
        elif self.opt.dataset == "cityscapes":
            self.model_vfi_train.load_state_dict(torch.load("./weights/IFRNet_L_CS.pth")["VFI"])
            self.model_vfi_test.load_state_dict(torch.load("./weights/IFRNet_S_CS.pth")["VFI"])
        else:
            pass
        self.model_vfi_train.to(self.device).eval()
        self.model_vfi_test.to(self.device).eval()

        if self.opt.world_size > 1:
            self.model_vfi_train = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_vfi_train)
            self.model_vfi_train = nn.parallel.DistributedDataParallel(self.model_vfi_train, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank, find_unused_parameters=True)
            self.model_vfi_test = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_vfi_test)
            self.model_vfi_test = nn.parallel.DistributedDataParallel(self.model_vfi_test, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank, find_unused_parameters=True)

        # optimizer settings
        if self.opt.optimizer == 'adamw':
            self.model_optimizer = torch.optim.AdamW(self.parameters_to_train, lr=self.opt.learning_rate, betas=(self.opt.beta1, self.opt.beta2),weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == 'adam':
            self.model_optimizer = torch.optim.Adam(self.parameters_to_train, lr=self.opt.learning_rate, betas=(self.opt.beta1, self.opt.beta2)) 
        elif self.opt.optimizer == 'sgd':
            self.model_optimizer = torch.optim.SGD(self.parameters_to_train, lr=self.opt.learning_rate, momentum=self.opt.momentum)
        else:
            logging.error("Optimizer '%s' not defined. Use (adamw|adam|sgd) instead", self.opt.optimizer)

        if self.opt.lr_sche_type == 'cos':
            self.model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, T_max=self.num_total_steps, eta_min=self.opt.eta_min)
        elif self.opt.lr_sche_type == 'step':
            self.model_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer, self.opt.decay_step, self.opt.decay_rate)
        if checkpoint:
            self.model_optimizer.load_state_dict(checkpoint["optimizer"])
            self.model_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            del checkpoint
        
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.opt.batch_size, self.opt.height, self.opt.width)
        self.project_3d.to(self.device)

        if self.opt.dataset == "kitti":
            logging.info("Using split: %s", self.opt.split)

        logging.info("There are {:d} training items and {:d} test items\n".format(len(train_dataset), len(test_dataset)))
        if self.opt.world_size > 1:
            dist.barrier()

    def set_seed(self, seed=1234):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        for self.epoch in range(self.ep_start, self.opt.num_epochs):
            self.run_epoch()
            if self.opt.lr_sche_type == "step":
                self.model_lr_scheduler.step()
            with torch.no_grad():
                if self.opt.dataset == "kitti":
                    self.test_kitti()
                    self.test_kitti_mf()
                elif self.opt.dataset == "cityscapes":
                    self.test_cityscapes()
                    self.test_cityscapes_mf()
                elif self.opt.dataset == "nyuv2":
                    self.test_nyuv2()
                else:
                    pass
            if self.opt.global_rank == 0:
                self.save_model(ep_end=True)

    def test_nyuv2(self):
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))

        self.set_eval()
        pred_depths = []
        gt_depths = []

        for idx, (color, depth) in enumerate(self.test_loader):
            input_color = color.to(self.device)
            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disp = pred_disp[:, 0]
            
            gt_depth = depth
            _, h, w = gt_depth.shape
            pred_depth = 1 / pred_disp
            pred_depth = F.interpolate(pred_depth.unsqueeze(0), (h, w), mode="nearest")[0]
            pred_depths.append(pred_depth)
            gt_depths.append(gt_depth)
        pred_depths = torch.cat(pred_depths, dim=0)
        gt_depths = torch.cat(gt_depths, dim=0).to(self.device)

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

        logging.info(" Mono evaluation - using median scaling")
        logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def test_cityscapes(self):
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()       

        pred_disps = []
        for idx, data in enumerate(self.test_loader):
            input_color = data[("color", 0, 0)].to(self.device)
            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i]).cuda()
            gt_height, gt_width = gt_depth.shape[:2]

            # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
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

            if self.opt.use_stereo:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio  
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = torch.tensor(ratios)
            med = torch.median(ratios)
            std = torch.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def test_kitti(self):
        """Test the model on a single minibatch
        """
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        # Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()

        pred_disps = []
        for idx, data in enumerate(self.test_loader):

            input_color = data[("color", 0, 0)].to(self.device)
            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i]).cuda()
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i:i+1].unsqueeze(0)
            pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=False)
            pred_depth = 1 / pred_disp[0, 0, :]
            if self.opt.eval_split == "eigen":
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
            if self.opt.use_stereo:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio  
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = torch.tensor(ratios)
            med = torch.median(ratios)
            std = torch.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train() 

    def test_cityscapes_mf(self):
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()       

        pred_disps = []
        for idx, data in enumerate(self.test_loader):
            img_n1 = data[("color", -1, 0)].to(self.device)
            img_p1 = data[("color", 1, 0)].to(self.device)
            img_0 = data[("color", 0, 0)].to(self.device)
            embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(self.device)
            embt = embt.repeat(img_n1.shape[0], 1, 1, 1)
            flow_0_n1, flow_0_p1, merge_mask_01 = self.model_vfi_test(img_n1, img_p1, embt, onlyFlow=True)

            feats_n1 = self.models["encoder_mf"](img_n1)
            feats_p1 = self.models["encoder_mf"](img_p1)
            feats_0 = self.models["encoder_mf"](img_0)
            feats = [feats_n1, feats_0, feats_p1]
            flows = [flow_0_n1, flow_0_p1]
            feats = self.models["fusion_module"](feats, flows, merge_mask_01)
            output = self.models["depth_mf"](feats)

            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i]).cuda()
            gt_height, gt_width = gt_depth.shape[:2]

            # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
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

            if self.opt.use_stereo:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio  
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = torch.tensor(ratios)
            med = torch.median(ratios)
            std = torch.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def test_kitti_mf(self):
        """Test the model on a single minibatch
        """
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        # Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()

        embt = torch.tensor(0.5).view(1, 1, 1, 1).float().cuda()
        pred_disps = []
        for idx, data in enumerate(self.test_loader):
            img_n1 = data[("color", -1, 0)].to(self.device)
            img_p1 = data[("color", 1, 0)].to(self.device)
            img_0 = data[("color", 0, 0)].to(self.device)
            embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(self.device)
            embt = embt.repeat(img_n1.shape[0], 1, 1, 1)
            flow_0_n1, flow_0_p1, merge_mask_01 = self.model_vfi_test(img_n1, img_p1, embt, onlyFlow=True)

            feats_n1 = self.models["encoder_mf"](img_n1)
            feats_p1 = self.models["encoder_mf"](img_p1)
            feats_0 = self.models["encoder_mf"](img_0)
            feats = [feats_n1, feats_0, feats_p1]
            flows = [flow_0_n1, flow_0_p1]
            feats = self.models["fusion_module"](feats, flows, merge_mask_01)
            output = self.models["depth_mf"](feats)

            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i]).cuda()
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i:i+1].unsqueeze(0)
            pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=False)
            pred_depth = 1 / pred_disp[0, 0, :]
            if self.opt.eval_split == "eigen":
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
            if self.opt.use_stereo:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio  
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = torch.tensor(ratios)
            med = torch.median(ratios)
            std = torch.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = torch.tensor(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        
        logging.info("Training epoch {}\n".format(self.epoch))

        self.sampler.set_epoch(self.epoch)
        self.sampler.set_start_iter(self.batch_start*self.opt.batch_size)
        self.set_train()

        if self.opt.world_size > 1:
            dist.barrier()
        start_data_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            self.step += 1
            start_fp_time = time.time()
            outputs, losses = self.process_batch(inputs)

            start_bp_time = time.time()
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            if self.opt.clip_grad != -1:
                for params in self.model_optimizer.param_groups:
                    params = params['params']
                    nn.utils.clip_grad_norm_(params, max_norm=self.opt.clip_grad)

            self.model_optimizer.step()

            if self.opt.lr_sche_type == "cos":
                self.model_lr_scheduler.step()

            # compute the process time
            data_time = start_fp_time - start_data_time
            fp_time = start_bp_time - start_fp_time
            bp_time = time.time() - start_bp_time

            # logging
            if ((batch_idx+self.batch_start) % self.opt.log_frequency == 0):
                if self.opt.world_size > 1:
                    dist.barrier()
                    for k in losses.keys():
                        dist.all_reduce(losses[k], op=dist.ReduceOp.SUM)
                        losses[k] /= self.opt.world_size
                    dist.barrier()
                if self.opt.global_rank == 0:
                    self.log_time(batch_idx+self.batch_start, data_time, fp_time,bp_time, losses["loss"].cpu().data)
                    self.log_tensorboard("train", losses)

            # save ckpt
            if ((batch_idx+self.batch_start)>0 and (batch_idx+self.batch_start) % self.opt.save_frequency == 0):
                if self.opt.global_rank == 0:
                    self.save_model(batch_idx=batch_idx+self.batch_start+1)
            if self.opt.world_size > 1:
                dist.barrier()
            start_data_time = time.time()

        self.batch_start = 0
     
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            try:
                inputs[key] = ipt.to(self.device)
            except:
                pass

        embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(self.device)
        embt = embt.repeat(self.opt.batch_size, 1, 1, 1)
        img_n1 = inputs[("color", -1, 0)]
        img_p1 = inputs[("color", 1, 0)]
        img_0 = inputs[("color", 0, 0)]

        ## n1 denotes -1 (negative),  nt denotes -t
        ## p1 denotes +1 (positive),  pt denotes +t
        with torch.no_grad():
            img_nt, flow_nt_n1, flow_nt_0, merge_mask_nt = self.model_vfi_train(img_n1, img_0, embt)
            img_pt, flow_pt_0, flow_pt_p1, merge_mask_pt = self.model_vfi_train(img_0, img_p1, embt)
            flow_0_n1, flow_0_p1, merge_mask_01 = self.model_vfi_train(img_n1, img_p1, embt, onlyFlow=True)     
     
        K = inputs[("K", 0)]
        inv_K = inputs[("inv_K", 0)]

        losses = {}

        losses["loss_base"] = torch.tensor(0.0).to(self.device)
        losses["loss_dc"] = torch.tensor(0.0).to(self.device)

        pose_n1_0, pose_0_n1 = self.predict_poses(inputs[("color_aug", -1, 0)], inputs[("color_aug", 0, 0)])
        pose_0_p1, pose_p1_0 = self.predict_poses(inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)])
        pose_n1_nt, pose_nt_n1 = self.predict_poses(img_n1, img_nt)
        pose_nt_p1, pose_p1_nt = self.predict_poses(img_nt, img_p1)
        pose_n1_pt, pose_pt_n1 = self.predict_poses(img_n1, img_pt)
        pose_pt_p1, pose_p1_pt = self.predict_poses(img_pt, img_p1)

        ## predict single-frame depths
        feats_0 = self.models["encoder"](inputs[("color_aug", 0, 0)])
        feats_nt = self.models["encoder"](img_nt)
        feats_pt = self.models["encoder"](img_pt)
        disp_0 = self.models["depth"](feats_0)
        disp_pt = self.models["depth"](feats_pt)   
        disp_nt = self.models["depth"](feats_nt)    
        _, depth_0 = disp_to_depth(disp_0[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        _, depth_pt = disp_to_depth(disp_pt[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        _, depth_nt = disp_to_depth(disp_nt[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

        ## calculate the self-supervised losses on single-frame depths
        img_n1_00 = self.generate_images_pred(disp_0, pose_0_n1, img_n1, K, inv_K)
        img_p1_00 = self.generate_images_pred(disp_0, pose_0_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_0, img_0, [img_n1_00, img_p1_00], [img_n1, img_p1])
        losses["loss_base"] += loss_base

        img_n1_pt = self.generate_images_pred(disp_pt, pose_pt_n1, img_n1, K, inv_K)
        img_p1_pt = self.generate_images_pred(disp_pt, pose_pt_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_pt, img_pt, [img_n1_pt, img_p1_pt], [img_n1, img_p1])
        losses["loss_base"] += loss_base  

        img_n1_nt = self.generate_images_pred(disp_nt, pose_nt_n1, img_n1, K, inv_K)
        img_p1_nt = self.generate_images_pred(disp_nt, pose_nt_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_nt, img_nt, [img_n1_nt, img_p1_nt], [img_n1, img_p1])
        losses["loss_base"] += loss_base  

        ## predict multi-frame depths
        if self.opt.fuse_model_type == "separate_all":
            feats_0 = self.models["encoder_mf"](inputs[("color_aug", 0, 0)])
            feats_nt = self.models["encoder_mf"](img_nt)
            feats_pt = self.models["encoder_mf"](img_pt)
            feats_n1 = self.models["encoder_mf"](inputs[("color_aug", -1, 0)])
            feats_p1 = self.models["encoder_mf"](inputs[("color_aug", 1, 0)])
        else:
            feats_n1 = self.models["encoder"](inputs[("color_aug", -1, 0)])
            feats_p1 = self.models["encoder"](inputs[("color_aug", 1, 0)])
        
        feats = [feats_n1, feats_0, feats_p1]
        flows = [flow_0_n1, flow_0_p1]
        feats = self.models["fusion_module"](feats, flows, merge_mask_01)
        disp_0_fuse = self.models["depth_mf"](feats)
        _, depth_0_fuse = disp_to_depth(disp_0_fuse[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

        feats = [feats_n1, feats_nt, feats_0]
        flows = [flow_nt_n1, flow_nt_0]
        feats = self.models["fusion_module"](feats, flows, merge_mask_nt)
        disp_nt_fuse = self.models["depth_mf"](feats)
        _, depth_nt_fuse = disp_to_depth(disp_nt_fuse[("disp", 0)], self.opt.min_depth, self.opt.max_depth)  

        feats = [feats_0, feats_pt, feats_p1]
        flows = [flow_pt_0, flow_pt_p1]
        feats = self.models["fusion_module"](feats, flows, merge_mask_pt)
        disp_pt_fuse = self.models["depth_mf"](feats)
        _, depth_pt_fuse = disp_to_depth(disp_pt_fuse[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        
        ## calculate the self-supervised losses on multi-frame depths
        ## and the depth consistency losses (SVDC)
        img_n1_0 = self.generate_images_pred(disp_0_fuse, pose_0_n1, img_n1, K, inv_K)
        img_p1_0 = self.generate_images_pred(disp_0_fuse, pose_0_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_0_fuse, img_0, [img_n1_0, img_p1_0], [img_n1, img_p1])
        losses["loss_base"] += loss_base
        loss_dc = self.compute_SI_log_depth_loss(depth_0, depth_0_fuse)
        losses["loss_dc"] += loss_dc

        img_n1_nt = self.generate_images_pred(disp_nt_fuse, pose_nt_n1, img_n1, K, inv_K)
        img_p1_nt = self.generate_images_pred(disp_nt_fuse, pose_nt_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_nt_fuse, img_nt, [img_n1_nt, img_p1_nt], [img_n1, img_p1])
        losses["loss_base"] += loss_base  
        loss_dc = self.compute_SI_log_depth_loss(depth_nt, depth_nt_fuse)
        losses["loss_dc"] += loss_dc
   
        img_n1_pt = self.generate_images_pred(disp_pt_fuse, pose_pt_n1, img_n1, K, inv_K)
        img_p1_pt = self.generate_images_pred(disp_pt_fuse, pose_pt_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_pt_fuse, img_pt, [img_n1_pt, img_p1_pt], [img_n1, img_p1])
        losses["loss_base"] += loss_base     
        loss_dc = self.compute_SI_log_depth_loss(depth_pt, depth_pt_fuse)
        losses["loss_dc"] += loss_dc     

        ## losses relevant to affine augmentation
        if self.opt.use_affine:           
            ## for img_0 
            disp_0_affine = self.models["depth"](self.models["encoder"](inputs[("color_affine_aug", 0, 0)]))
            _, depth_0_affine = disp_to_depth(disp_0_affine[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            Rc = inputs[("Rc")]
            Rt_Rc = torch.zeros_like(pose_0_n1).to(self.device)
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(pose_0_n1[:, :3, :3], torch.inverse(Rc)))
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, pose_0_n1[:, :3, 3:4])  
            pose_0_n1_affine = Rt_Rc

            Rt_Rc = torch.zeros_like(pose_0_p1).to(self.device)
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(pose_0_p1[:, :3, :3], torch.inverse(Rc)))
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, pose_0_p1[:, :3, 3:4])  
            pose_0_p1_affine = Rt_Rc

            img_n1_affine = inputs[("color_affine", -1, 0)]
            img_p1_affine = inputs[("color_affine", 1, 0)]
            img_0_affine = inputs[("color_affine", 0, 0)]
            mask_rec = inputs[("valid_mask_rec")]

            img_n1_0_affine = self.generate_images_pred(disp_0_affine, pose_0_n1_affine, img_n1_affine, K, inv_K)
            img_p1_0_affine = self.generate_images_pred(disp_0_affine, pose_0_p1_affine, img_p1_affine, K, inv_K)
            loss_base, _ = self.compute_losses_base(disp_0_affine, img_0_affine, [img_n1_0_affine, img_p1_0_affine], [img_n1_affine, img_p1_affine], mask_rec)
            ## calculate the self-supervised loss on augmented single-frame depth
            losses["loss_base"] += loss_base
            ##  calculate two scale-aware depth consistency losses (SADC)
            losses["loss_dc"] += self.compute_depth_consistency_loss_affine(depth_0_affine, depth_0, depth_0_fuse, inputs)  

            ## for img_nt
            img_nt_affine = self.affine_transform(img_nt, inputs)
            disp_nt_affine = self.models["depth"](self.models["encoder"](img_nt_affine))
            _, depth_nt_affine = disp_to_depth(disp_nt_affine[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            Rc = inputs[("Rc")]
            Rt_Rc = torch.zeros_like(pose_nt_n1).to(self.device)
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(pose_nt_n1[:, :3, :3], torch.inverse(Rc)))
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, pose_nt_n1[:, :3, 3:4])  
            pose_nt_n1_affine = Rt_Rc

            Rt_Rc = torch.zeros_like(pose_nt_p1).to(self.device)
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(pose_nt_p1[:, :3, :3], torch.inverse(Rc)))
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, pose_nt_p1[:, :3, 3:4])  
            pose_nt_p1_affine = Rt_Rc

            img_n1_nt_affine = self.generate_images_pred(disp_nt_affine, pose_nt_n1_affine, img_n1_affine, K, inv_K)
            img_p1_nt_affine = self.generate_images_pred(disp_nt_affine, pose_nt_p1_affine, img_p1_affine, K, inv_K)
            loss_base, _ = self.compute_losses_base(disp_nt_affine, img_nt_affine, [img_n1_nt_affine, img_p1_nt_affine], [img_n1_affine, img_p1_affine], mask_rec)
            losses["loss_base"] += loss_base  
            losses["loss_dc"] += self.compute_depth_consistency_loss_affine(depth_nt_affine, depth_nt, depth_nt_fuse, inputs)

            ## for img_pt
            img_pt_affine = self.affine_transform(img_pt, inputs)
            disp_pt_affine = self.models["depth"](self.models["encoder"](img_pt_affine))
            _, depth_pt_affine = disp_to_depth(disp_pt_affine[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            Rc = inputs[("Rc")]
            Rt_Rc = torch.zeros_like(pose_pt_n1).to(self.device)
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(pose_pt_n1[:, :3, :3], torch.inverse(Rc)))
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, pose_pt_n1[:, :3, 3:4])  
            pose_pt_n1_affine = Rt_Rc

            Rt_Rc = torch.zeros_like(pose_pt_p1).to(self.device)
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(pose_pt_p1[:, :3, :3], torch.inverse(Rc)))
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, pose_pt_p1[:, :3, 3:4])  
            pose_pt_p1_affine = Rt_Rc

            img_n1_pt_affine = self.generate_images_pred(disp_pt_affine, pose_pt_n1_affine, img_n1_affine, K, inv_K)
            img_p1_pt_affine = self.generate_images_pred(disp_pt_affine, pose_pt_p1_affine, img_p1_affine, K, inv_K)
            loss_base, _ = self.compute_losses_base(disp_pt_affine, img_pt_affine, [img_n1_pt_affine, img_p1_pt_affine], [img_n1_affine, img_p1_affine], mask_rec)
            losses["loss_base"] += loss_base  
            losses["loss_dc"] += self.compute_depth_consistency_loss_affine(depth_pt_affine, depth_pt, depth_pt_fuse, inputs)

        losses["loss"] = losses["loss_base"] + self.opt.lamda * losses["loss_dc"]
        return None, losses

    def affine_transform(self, img, inputs):
        # img: tensor [B, 3, H, W]
        img_affine = []
        for b in range(self.opt.batch_size):
            angle = inputs[("angle")][b][0].item()
            x0 = inputs[("box")][b, 0].item()
            y0 = inputs[("box")][b, 1].item()  
            w = inputs[("box")][b, 2].item()
            h = inputs[("box")][b, 3].item()   
            img_b = img[b].unsqueeze(0)
            img_b = transforms.functional.rotate(img_b, angle=angle, interpolation=2)
            img_b = img_b[:, :, y0:y0+h, x0:x0+w]
            img_b = torch.nn.functional.interpolate(img_b, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            img_affine.append(img_b)
        return torch.cat(img_affine, 0)

    def compute_depth_consistency_loss_affine(self, depth_affine, depth, depth_fuse, inputs):
        loss_dc_affine = 0
        for b in range(self.opt.batch_size):
            angle = inputs[("angle")][b][0].item()
            x0 = inputs[("box")][b, 0].item()
            y0 = inputs[("box")][b, 1].item()  
            w = inputs[("box")][b, 2].item()
            h = inputs[("box")][b, 3].item()
            tmp = F.interpolate(depth_affine[b].unsqueeze(0), [h, w], mode="bilinear", align_corners=False)
            depth_restore = torch.zeros((1, 1, self.opt.height, self.opt.width)).to(self.device)
            depth_restore[:, :, y0:y0+h, x0:x0+w] = tmp
            depth_restore = transforms.functional.rotate(depth_restore, angle=-angle, interpolation=2)
            depth_restore *= inputs[("ratio_local")][b, 0]
            depth_origin_fuse = depth_fuse[b].unsqueeze(0)
            depth_origin = depth[b].unsqueeze(0)
            loss_dc_affine += self.compute_SI_log_depth_loss(depth_restore, depth_origin_fuse, inputs[("valid_mask_cons")][b].unsqueeze(0))
            loss_dc_affine += self.compute_SI_log_depth_loss(depth_restore, depth_origin, inputs[("valid_mask_cons")][b].unsqueeze(0))
        loss_dc_affine /= self.opt.batch_size
        return loss_dc_affine
  
    def compute_SI_log_depth_loss(self, pred, target, mask=None, beta=0.5):
        # B*1*H*W  ->  B*H*W
        if mask is None:
            mask = torch.ones_like(pred).to(self.device)

        mask = mask[:, 0] 
        log_pred = torch.log(pred[:, 0]+1e-7) * mask
        log_tgt = torch.log(target[:, 0]+1e-7) * mask

        log_diff = log_pred - log_tgt
        valid_num = mask.sum(1).sum(1) + 1e-8
        log_diff_squre_sum = (log_diff ** 2).sum(1).sum(1)
        log_diff_sum_squre = (log_diff.sum(1).sum(1)) ** 2
        loss = log_diff_squre_sum/valid_num - beta*log_diff_sum_squre/(valid_num**2)

        loss = loss.mean()

        return loss

    def predict_poses(self, img_0, img1):
        """Predict poses between input frames for monocular sequences.
        """

        pose_inputs = [img_0, img1]
        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
        
        axisangle, translation = self.models["pose"](pose_inputs)

        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)
        pose_inv = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        return pose, pose_inv
    
    def generate_images_pred(self, disp_tgt, pose_tgt_src, img_src, K, inv_K):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        disp = disp_tgt[("disp", 0)]
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        cam_points = self.backproject_depth(depth, inv_K)
        pix_coords = self.project_3d(cam_points, K, pose_tgt_src)

        img_src_tgt = F.grid_sample(
            img_src,
            pix_coords,
            padding_mode="border", align_corners=True)
    
        return img_src_tgt

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses_base(self, disp_tgt, img_tgt, imgs_src_tgt, imgs_src, mask_rec=None):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        loss = 0
        reprojection_losses = []

        disp = disp_tgt[("disp", 0)]

        for i in range(len(imgs_src_tgt)):
            pred = imgs_src_tgt[i]
            reprojection_losses.append(self.compute_reprojection_loss(pred, img_tgt))

        reprojection_losses = torch.cat(reprojection_losses, 1)
        
        if not self.opt.disable_automasking:
            identity_reprojection_losses = []
            for i in range(len(imgs_src)):
                pred = imgs_src[i]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, img_tgt))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)
        
        if mask_rec is not None:
            to_optimise *= mask_rec[:,0]

        if not self.opt.disable_automasking:
            auto_mask = (idxs > identity_reprojection_loss.shape[1] - 1).float().unsqueeze(1)
        else:
            auto_mask = None

        loss += to_optimise.mean()
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)

        smooth_loss = get_smooth_loss(norm_disp, img_tgt)

        loss += self.opt.disparity_smoothness * smooth_loss

        return loss, auto_mask

    def log_time(self, batch_idx, data_time, fp_time, bp_time, loss):
        """Print a logging statement to the terminal
        """
        batch_time = data_time + fp_time + bp_time
        # time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps - self.step) * batch_time if self.step > 1 else 0
        print_string = "epoch: {:>2}/{} | batch: {:>4}/{} | data time: {:.4f}" + " | batch time: {:.3f} | loss: {:.4f} | lr: {:.2e} | time left: {}"
        logging.info(print_string.format(self.epoch, self.opt.num_epochs-1,batch_idx, self.num_steps_per_epoch, data_time, batch_time, loss, self.model_optimizer.state_dict()['param_groups'][0]['lr'], sec_to_hm_str(training_time_left)))

    def log_tensorboard(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        #     for s in range(self.opt.num_scales):
        #         for frame_id in self.opt.frame_ids:
        #             writer.add_image(
        #                 "color_{}_{}/{}".format(frame_id, s, j),
        #                 inputs[("color", frame_id, s)][j].data, self.step)
        #             if s == 0 and frame_id != 0:
        #                 writer.add_image(
        #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                     outputs[("color", frame_id, s)][j].data, self.step)

        #         writer.add_image(
        #             "disp_{}/{}".format(s, j),
        #             normalize_image(outputs[("disp", s)][j]), self.step)

        #         if not self.opt.disable_automasking:
        #             writer.add_image(
        #                 "automask_{}/{}".format(s, j),
        #                 outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(self.log_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
        source_folder = os.path.split(os.path.realpath(__file__))[0]+'/'
        target_folder = os.path.join(self.log_path, 'codes')
        os.system("rm -rf {}".format(target_folder))
        exts = [".sh", ".py"] 
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if any(file.endswith(ext) for ext in exts):
                    source_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(source_file_path, source_folder)
                    target_file_path = os.path.join(target_folder, relative_path)
                    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                    shutil.copy(source_file_path, target_file_path)
                    
    def save_model(self, ep_end=False, batch_idx=0):
        """Save model weights to disk
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = {}
        for model_name, model in self.models.items():
            if self.opt.world_size == 1:
                to_save[model_name] = model.state_dict()
            else:
                to_save[model_name] = model.module.state_dict()
        to_save['height'] = self.opt.height
        to_save['width'] = self.opt.width
        to_save['use_stereo'] = self.opt.use_stereo   
        if ep_end:
            save_ep_path = os.path.join(models_dir, "model_{}.pth".format(self.epoch))
            torch.save(to_save, save_ep_path)  ## only save the model weights
            to_save["epoch"] = self.epoch + 1
        else:
            to_save["epoch"] = self.epoch

        to_save['step_in_total'] = self.step
        to_save["batch_idx"] = batch_idx
        to_save['optimizer'] = self.model_optimizer.state_dict()
        to_save['lr_scheduler'] = self.model_lr_scheduler.state_dict()
        
        save_path = os.path.join(self.log_path, "ckpt.pth")
        torch.save(to_save, save_path)

    def load_ckpt(self):
        """Load checkpoint to resume a training, used in training process.
        """
        logging.info(" ")
        load_path = os.path.join(self.log_path, "ckpt.pth")
        if not os.path.exists(load_path):
            logging.info("No checkpoint to resume, train from epoch 0.")
            return None

        logging.info("Resume checkpoint from {}".format(os.path.abspath(load_path)))
        checkpoint = torch.load(load_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.ep_start = checkpoint['epoch']
        self.batch_start = checkpoint['batch_idx']
        self.step = checkpoint['step_in_total']
        logging.info("Start at eopch {}, batch index {}".format(self.ep_start, self.batch_start))
        return checkpoint

    def load_pretrained_model(self):
        """Load pretrained model(s) from disk, used for initializing.
        """
        self.opt.pretrained_path = os.path.abspath(self.opt.pretrained_path)

        assert os.path.exists(self.opt.pretrained_path), \
            "Cannot find folder {}".format(self.opt.pretrained_path)
        logging.info("Loading pretrained model from folder: {}".format(self.opt.pretrained_path))

        checkpoint = torch.load(self.opt.pretrained_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

if __name__ == "__main__":
    opts.world_size = torch.cuda.device_count()
    if opts.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opts.local_rank)
        opts.global_rank = torch.distributed.get_rank()
    trainer = Trainer(opts)
    trainer.train()