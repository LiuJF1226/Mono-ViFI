from .kitti_dataset import KITTIRAWDataset, KITTIOdomDataset, KITTIDepthDataset
from .make3d_dataset import Make3DDataset
from .nyuv2_dataset import NYUDataset
from .cityscapes_dataset import CityscapesDataset
from .VFI_dataset import KITTI_VFI_Dataset, Cityscapes_VFI_Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
import torch

class CustomSampler(Sampler):
    def __init__(self, dataset, seed=0):
        self.len = len(dataset)
        self.start_iter = 0
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.len, generator=g).tolist()
        indices = indices[self.start_iter:]
        return iter(indices)
    
    def __len__(self):
        return self.len

    def set_start_iter(self, start_iter):
        self.start_iter = start_iter

    def set_epoch(self, epoch):
        self.epoch = epoch


class CustomDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.
        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas
        # self.num_samples = self.total_size 
        # logging.info(f"rank: {self.rank}: Sampler created...")
        # logging.info(f"sample_num: {len(dataset) * self.num_replicas}")
        # print(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
                
        # resume the sampler
        indices = indices[self.start_iter:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter