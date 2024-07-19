from __future__ import division
import argparse
import PIL.Image as pil
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dump_root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()


class cityscapes_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split='train',
                 crop_bottom=True, # Get rid of the car logo
                 sample_gap=2,  # Sample every two frames to match KITTI frame rate
                 img_height=171, 
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.split = split
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = crop_bottom
        self.sample_gap = sample_gap
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        assert seq_length % 2 != 0, 'seq_length must be odd!'
        self.frames = self.collect_frames(split)
        self.num_frames = len(self.frames)
        if split == 'train':
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print('Total frames collected: %d' % self.num_frames)
        
    def collect_frames(self, split):
        img_dir = self.dataset_dir + '/leftImg8bit_sequence/' + split + '/'
        city_list = os.listdir(img_dir)
        frames = []
        for city in city_list:
            img_files = glob(img_dir + city + '/*.png')
            for f in img_files:
                frame_id = os.path.basename(f).split('leftImg8bit')[0]
                frames.append(frame_id)
        return frames

    def get_train_example_with_idx(self, tgt_idx):
        tgt_frame_id = self.frames[tgt_idx]
        if not self.is_valid_example(tgt_frame_id):
            return False
        example = self.load_example(self.frames[tgt_idx])
        return example

    def load_intrinsics(self, frame_id, split):
        city, seq, _, _ = frame_id.split('_')
        camera_file = os.path.join(self.dataset_dir, 'camera',
                                   split, city, city + '_' + seq + '_*_camera.json')
        camera_file = glob(camera_file)[0]
        with open(camera_file, 'r') as f: 
            camera = json.load(f)
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
        intrinsics = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0,  0,  1]])
        return intrinsics

    def is_valid_example(self, tgt_frame_id):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence', 
                                self.split, city, curr_frame_id + 'leftImg8bit.png')
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_frame_id, seq_length, crop_bottom):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence', 
                                self.split, city, curr_frame_id + 'leftImg8bit.png')
            curr_img = pil.open(curr_image_file).convert('RGB')
            raw_shape = np.copy(curr_img.size)
            if o == 0:
                zoom_y = self.img_height/raw_shape[1]
                zoom_x = self.img_width/raw_shape[0]
            curr_img = curr_img.resize((self.img_width, self.img_height), pil.LANCZOS)
            if crop_bottom:
                ymax = int(curr_img.size[1] * 0.75)
                curr_img = curr_img.crop((0, 0, curr_img.size[0], ymax))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y
    
    def load_example(self, tgt_frame_id, load_gt_pose=False):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_frame_id, self.seq_length, self.crop_bottom)
        intrinsics = self.load_intrinsics(tgt_frame_id, self.split)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_frame_id.split('_')[0]
        example['file_name'] = tgt_frame_id[:-1]
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out
    

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = np.array(im)
        else:
            im = np.array(im)
            res = np.concatenate((res, im))
    res = pil.fromarray(res)
    return res

def dump_example(n, args):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.png' % example['file_name']
    image_seq.save(dump_img_file)
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader

    data_loader = cityscapes_loader(args.dataset_dir,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))

    # # Split into train/val
    # np.random.seed(8964)
    # subfolders = os.listdir(args.dump_root)
    # with open(args.dump_root + 'train.txt', 'w') as tf:
    #     with open(args.dump_root + 'val.txt', 'w') as vf:
    #         for s in subfolders:
    #             if not os.path.isdir(args.dump_root + '/%s' % s):c
    #                 continue
    #             imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
    #             frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
    #             for frame in frame_ids:
    #                 if np.random.random() < 0.1:
    #                     vf.write('%s %s\n' % (s, frame))
    #                 else:
    #                     tf.write('%s %s\n' % (s, frame))

main()