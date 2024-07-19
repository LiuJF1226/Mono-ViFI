import numpy as np
import time
import logging

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
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
from flow_vis import flow_to_color
import datasets
from networks import *
import shutil


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

        self.ep_start = 0
        self.batch_start = 0
        self.step = 0

        ## now only support for KITTI and Cityscapes
        datasets_dict = {"kitti": datasets.KITTI_VFI_Dataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes": datasets.Cityscapes_VFI_Dataset,
                         "nyuv2": datasets.NYUDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset == "kitti":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.split, "{}_files.txt")
        elif self.opt.dataset == "kitti_odom":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", "odom", "{}_files.txt")
        elif self.opt.dataset == "nyuv2":
            fpath = os.path.join(os.path.dirname(__file__), "splits/nyuv2", "{}_files.txt")
        elif self.opt.dataset == "cityscapes":
            fpath = os.path.join(os.path.dirname(__file__), "splits/cityscapes", "{}_files.txt") 
        else:
            pass

        train_filenames = readlines(fpath.format("train"))
        img_ext = '.jpg' if self.opt.jpg else '.png'

        num_train_samples = len(train_filenames)
        self.num_steps_per_epoch = num_train_samples // self.opt.world_size // self.opt.batch_size
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, is_train=True, img_ext=img_ext)
        if self.opt.world_size > 1:
            self.sampler = datasets.CustomDistributedSampler(train_dataset, self.opt.seed)
        else:
            self.sampler = datasets.CustomSampler(train_dataset, self.opt.seed)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False, sampler=self.sampler, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        # create models
        self.models["VFI"] = IFRNet(scale=self.opt.vfi_scale)
        self.models["VFI"].to(self.device)
        if self.opt.pretrained_path:
            logging.info("Loading pretrained model from folder: {}".format(self.opt.pretrained_path))
            try:
                self.models["VFI"].load_state_dict(torch.load(self.opt.pretrained_path)["VFI"])
            except:
                self.models["VFI"].load_state_dict(torch.load(self.opt.pretrained_path))

        if self.opt.resume:
            checkpoint = self.load_ckpt()

        for k in self.models.keys():
            self.parameters_to_train = self.parameters_to_train + list(self.models[k].parameters())

        if self.opt.world_size > 1:
            for k in self.models.keys():
                self.models[k] = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.models[k])
                self.models[k] = nn.parallel.DistributedDataParallel(self.models[k], device_ids=[self.opt.local_rank], output_device=self.opt.local_rank, find_unused_parameters=True)
  
        # optimizer settings
        if self.opt.optimizer == 'adamw':
            self.model_optimizer = torch.optim.AdamW(self.parameters_to_train,lr=self.opt.learning_rate, betas=(self.opt.beta1, self.opt.beta2),weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == 'adam':
            self.model_optimizer = torch.optim.Adam(self.parameters_to_train,lr=self.opt.learning_rate, betas=(self.opt.beta1, self.opt.beta2)) 
        elif self.opt.optimizer == 'sgd':
            self.model_optimizer = torch.optim.SGD(self.parameters_to_train,lr=self.opt.learning_rate, momentum=self.opt.momentum)
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

        if self.opt.dataset == "kitti":
            logging.info("Using split: %s", self.opt.split)
        logging.info("There are {:d} training items \n".format(len(train_dataset)))
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
            if self.opt.global_rank == 0:
                self.save_model(ep_end=True)

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
            for l in range(len(inputs)):
                inputs[l] = inputs[l].to(self.device)
            img0, img1, img2, embt = inputs

            self.step += 1
            start_fp_time = time.time()

            img1_pred, loss_vfi, flow0, flow1, _ = self.models['VFI'](img0, img2, embt, img1)
           
            start_bp_time = time.time()
            self.model_optimizer.zero_grad()

            losses = {}
            losses["loss"] = loss_vfi
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

            # # logging
            if ((batch_idx+self.batch_start) % self.opt.log_frequency == 0):
                if self.opt.world_size > 1:
                    dist.barrier()
                    for k in losses.keys():
                        dist.all_reduce(losses[k], op=dist.ReduceOp.SUM)
                        losses[k] = losses[k] / self.opt.world_size
                    dist.barrier()
                if self.opt.global_rank == 0:
                    self.log_time(batch_idx+self.batch_start, data_time, fp_time, bp_time, losses["loss"].cpu().data)
                    self.log_tensorboard("train", img0, img1, img2, img1_pred, flow0, flow1, losses)

            # save ckpt
            if ((batch_idx+self.batch_start)>0 and (batch_idx+self.batch_start) % self.opt.save_frequency == 0):
                if self.opt.global_rank == 0:
                    self.save_model(batch_idx=batch_idx+self.batch_start+1)
            if self.opt.world_size > 1:
                dist.barrier()
            start_data_time = time.time()

        self.batch_start = 0

    def log_time(self, batch_idx, data_time, fp_time, bp_time, loss):
        """Print a logging statement to the terminal
        """
        batch_time = data_time + fp_time + bp_time
        # time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps - self.step) * batch_time if self.step > 1 else 0
        print_string = "epoch: {:>2}/{} | batch: {:>4}/{} | data time: {:.4f}" + " | fp time: {:.3f} | bp time: {:.3f} | loss: {:.4f} | lr: {:.2e} | time left: {}"
        logging.info(print_string.format(self.epoch, self.opt.num_epochs-1,batch_idx, self.num_steps_per_epoch, data_time, fp_time, bp_time, loss, self.model_optimizer.state_dict()['param_groups'][0]['lr'], sec_to_hm_str(training_time_left)))

    def log_tensorboard(self, mode, img0, img1, img2, img1_pred, flow0, flow1, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(3, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image("img0/{}".format(j), img0[j].data, self.step)
            writer.add_image("img1/{}".format(j), img1[j].data, self.step)
            writer.add_image("img2/{}".format(j), img2[j].data, self.step)
            writer.add_image("img1_pred/{}".format(j), img1_pred[j].data, self.step)
            f0 = flow0[j].cpu().data.permute(1, 2, 0).numpy()
            f0 = flow_to_color(f0, convert_to_bgr=True)
            writer.add_image("flow0/{}".format(j), torch.tensor(f0.transpose((2, 0, 1))), self.step)
            f1 = flow1[j].cpu().data.permute(1, 2, 0).numpy()
            f1 = flow_to_color(f1, convert_to_bgr=True)
            writer.add_image("flow1/{}".format(j), torch.tensor(f1.transpose((2, 0, 1))), self.step)

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
        to_save['vfi_scale'] = self.opt.vfi_scale 

        if ep_end:
            save_ep_path = os.path.join(models_dir, "model{}.pth".format(self.epoch))
            torch.save(to_save, save_ep_path)  ## only save the model weights after each epoch
            to_save["epoch"] = self.epoch + 1
        else:
            to_save["epoch"] = self.epoch 

        to_save['step_in_total'] = self.step
        to_save["batch_idx"] = batch_idx
        to_save['optimizer'] = self.model_optimizer.state_dict()
        to_save['lr_scheduler'] = self.model_lr_scheduler.state_dict()

        save_path = os.path.join(self.log_path, "ckpt.pth")
        torch.save(to_save, save_path)  ## also save the optimizer state for resuming

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


if __name__ == "__main__":
    opts.world_size = torch.cuda.device_count()
    if opts.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opts.local_rank)
        opts.global_rank = torch.distributed.get_rank()
    trainer = Trainer(opts)
    trainer.train()