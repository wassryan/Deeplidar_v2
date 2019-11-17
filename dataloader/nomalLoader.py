# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.dirname(ROOT_DIR)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    '''
    filepath: /nfs-data/zhengk_data/normal_data/workspace
    输出：RGB,sparse_normal,gt_normal
    :param filepath:
    :return:
    '''
    root_d = "/nfs-data/zhengk_data/normal_data/workspace/kitti_depth" # '/nfs-data/zhengk_data/normal_data/workspace/data/kitti_depth'
    root_d2 = "/home/zhengk/data/kitti_depth"
    root_rgb = "/home/zhengk/data/kitti_rgb" # 'data/kitti_rgb'

    glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
    pattern_d = ("groundtruth","velodyne_raw")

    def get_rgb_paths(p):
        ps = p.split('/')
        pnew = '/'.join([root_rgb] + ps[-6:-4] + ps[-2:-1] + ['data'] + ps[-1:])  # 'data/kitti_rgb' + '/train/*_sync/image_0*/data/*.png'
        return pnew
    def get_lidepth_paths(p):
        ps = p.split('/')
        pnew = '/'.join([root_d2] + ps[-5:-3] + ['proj_depth'] + ['velodyne_raw'] + ps[-3:-2] + ps[-1:])
        return pnew

    if glob_gt is not None: # "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        glob_gt = os.path.join(root_d,glob_gt) # '/nfs-data/.../workspace/data/kitti_depth' + glob_gt,
        paths_gt = sorted(glob.glob(glob_gt)) #/nfs-data/zhengk_data/normal_data/workspace/kitti_depth/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/*.png
        # paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt] # 雷达产生depth的图 train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png
        # 虽然Completion的train中的图是-+5 ID开始的，但是读取rgb图时是以completion中的图片路径替换字符串得到的，所以rgb和gt/depth是对齐的
        # print(paths_gt[0])
        paths_rgb = [get_rgb_paths(p) for p in paths_gt] # /home/zhengk/data/kitti_rgb/train/2011_10_03_drive_0042_sync/image_02/data/*.png
        paths_d = [ get_lidepth_paths(p) for p in paths_rgb ]
        print(paths_d[0])

    else:
        print("glob_gt is none!!")

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise(RuntimeError("Found 0 images in data folders"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise(RuntimeError("Produced different sizes for datasets"))

    return paths_rgb,paths_d,paths_gt # rgb image/lidar depth/surface normal(depth gt)


if __name__ == '__main__':
    datapath = ''
