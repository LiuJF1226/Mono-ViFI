import os
# import argparse
import configargparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

# parser = argparse.ArgumentParser(description="Mono-ViFI options")
parser = configargparse.ArgumentParser(description="Mono-ViFI options")

parser.add_argument('-c', '--config', required=True, is_config_file=True, 
                    help='config file path')

# Distributed Training. Do not modify!
parser.add_argument('--local_rank', 
                    default=0, 
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--global_rank', 
                    default=0, 
                    type=int,
                    help='global node rank for distributed training')
parser.add_argument('--world_size', 
                    default=1, 
                    type=int,
                    help='GPU numbers')

# PATHS
parser.add_argument("--data_path",
                    type=str,
                    help="path to the training data",
                    default=os.path.join(file_dir, "kitti_data"))
parser.add_argument("--data_path_pre",
                    type=str,
                    help="path to the preprocessed training data, specify when training on Cityscapes.")
parser.add_argument("--log_dir",
                    type=str,
                    help="log directory",
                    default=os.path.join(os.path.expanduser("~"), "tmp"))

# TRAINING options
parser.add_argument("--exp_name",
                    type=str,
                    help="the name of the folder to save the model in",
                    default="mdp")
parser.add_argument("--split",
                    type=str,
                    help="which training split to use, only for kitti",
                    choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                    default="eigen_zhou")
parser.add_argument("--eval_split",
                    type=str,
                    default="eigen",
                    choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                    help="which split to run eval on")
parser.add_argument("--num_layers",
                    type=int,
                    help="number of resnet layers",
                    default=18,
                    choices=[18, 34, 50, 101, 152])
parser.add_argument("--dataset",
                    type=str,
                    help="dataset to train on",
                    default="kitti",
                    choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "nyuv2", "cityscapes"])
parser.add_argument("--jpg",
                    help="if set, trains from jpg files (instead of pngs)",
                    action="store_true")
parser.add_argument("--height",
                    type=int,
                    help="input image height",
                    default=192)
parser.add_argument("--width",
                    type=int,
                    help="input image width",
                    default=640)
parser.add_argument("--disparity_smoothness",
                    type=float,
                    help="disparity smoothness weight",
                    default=1e-3)
parser.add_argument("--num_scales",
                    type=int,
                    help="scale numbers used in the loss",
                    default=1)
parser.add_argument("--min_depth",
                    type=float,
                    help="minimum depth",
                    default=0.1)
parser.add_argument("--max_depth",
                    type=float,
                    help="maximum depth",
                    default=100.0)
parser.add_argument("--lamda",
                    type=float,
                    help="the coefficient of depth consistency loss",
                    default=0.2)
parser.add_argument("--use_stereo",
                    help="if set, uses stereo pair for training",
                    action="store_true")
parser.add_argument("--frame_ids",
                    nargs="+",
                    type=int,
                    help="frames to load",
                    default=[0, -1, 1])

# OPTIMIZATION options
parser.add_argument('--optimizer',
                    type=str,
                    default='adamw',
                    help='choose optimizer, (default:adamw)',
                    choices=["adamw", "adam", "sgd"])
parser.add_argument('--lr_sche_type',
                    type=str,
                    default='step',
                    help='choose lr_scheduler, (default:cos)',
                    choices=["cos", "step"])
parser.add_argument('--eta_min',
                    type=float,
                    default=5e-6,
                    help='eta_min is the final learning rate when using cosine annealing.')
parser.add_argument("--batch_size",
                    type=int,
                    help="batch size",
                    default=12)
parser.add_argument('--learning_rate',
                    default=0.0001,
                    type=float,
                    help='learning rate')
parser.add_argument('--decay_rate',
                    dest='decay_rate',
                    type=float,
                    default=0.1,
                    help='# of optimizer')
parser.add_argument('--decay_step',
                    dest='decay_step',
                    type=int,
                    nargs='+',
                    default=[15],
                    help='# of optimizer')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.01,
                    help='Weight decay for adamw.')
parser.add_argument('--beta1',
                    type=float,
                    default=0.9,
                    help='beta1 for adamw.')
parser.add_argument('--beta2',
                    type=float,
                    default=0.999,
                    help='beta2 for adamw.')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum for sgd')
parser.add_argument('--clip_grad',
                    dest='clip_grad',
                    type=float,
                    default=5,
                    help='# of the parameters')
parser.add_argument("--num_epochs",
                    type=int,
                    help="number of epochs",
                    default=20)
parser.add_argument("--seed",
                    type=int,
                    help="seed for random",
                    default=1234)
parser.add_argument("--resume",
                    help="if set, try to resume from a exsited ckpt",
                    action="store_true")

# ABLATION options
parser.add_argument("--avg_reprojection",
                    help="if set, uses average reprojection loss",
                    action="store_true")
parser.add_argument("--disable_automasking",
                    help="if set, doesn't do auto-masking",
                    action="store_true")
parser.add_argument("--no_ssim",
                    help="if set, disables ssim in the loss",
                    action="store_true")
parser.add_argument("--weights_init",
                    type=str,
                    help="pretrained or scratch",
                    default="pretrained",
                    choices=["pretrained", "scratch"])
parser.add_argument("--backbone",
                    type=str,
                    default="ResNet18",
                    choices=["ResNet18", "ResNet50", "LiteMono", "DHRNet"])
parser.add_argument("--vfi_scale",
                    type=str,
                    help="the scale of IFRNet",
                    default="small",
                    choices=["large", "small"])
parser.add_argument("--fuse_model_type",
                    type=str,
                    help="separate or shared encoders/decoders",
                    default="shared_encoder",
                    choices=["shared_encoder", "separate_all", "shared_all"])
parser.add_argument("--use_affine",
                    help="set to use affine augmentation",
                    action="store_true")

# SYSTEM options
parser.add_argument("--num_workers",
                    type=int,
                    help="number of dataloader workers",
                    default=16)

# LOADING options
parser.add_argument("--pretrained_path",
                    type=str,
                    help="path of pretrained model to load")

# LOGGING options
parser.add_argument("--log_frequency",
                    type=int,
                    help="number of batches between each log",
                    default=500)
parser.add_argument("--save_frequency",
                    type=int,
                    help="number of epochs between each save",
                    default=500)

opts = parser.parse_args()