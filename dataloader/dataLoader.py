# -*- coding: utf-8 -*-
import os
import os.path
import glob
import numpy as np

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
    root_d = os.path.join(filepath, 'kitti_depth') # '/home/zhengk/data/kitti_depth'
    root_rgb = os.path.join(filepath, 'kitti_rgb') # '/home/zhengk/data/kitti_rgb'
    '''
    root_d = "/home/zhengk/data/kitti_depth" # '/home/zhengk/data/kitti_depth'
    root_rgb = "/home/zhengk/data/kitti_rgb" # '/home/zhengk/data/kitti_rgb'

    glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
    pattern_d = ("groundtruth","velodyne_raw")

    def get_rgb_paths(p):
        ps = p.split('/')
        pnew = '/'.join([root_rgb] + ps[-6:-4] + ps[-2:-1] + ['data'] + ps[-1:])  # 'data/kitti_rgb' + '/train/*_sync/image_0*/data/*.png'
        return pnew

    if glob_gt is not None: # "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        glob_gt = os.path.join(root_d,glob_gt) # 'data/kitti_depth' + glob_gt
        paths_gt = sorted(glob.glob(glob_gt))# 返回depth_gt的路径到列表中
        paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt] # 雷达产生depth的图 train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png
        # 虽然Completion的train中的图是-+5 ID开始的，但是读取rgb图时是以completion中的图片路径替换字符串得到的，所以rgb和gt/depth是对齐的
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]

    else:
        print("glob_gt is none!!")

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise(RuntimeError("Found 0 images in data folders"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise(RuntimeError("Produced different sizes for datasets"))

    return paths_rgb,paths_d,paths_gt # rgb/sparse depth/gt depth

if __name__ == '__main__':
    datapath = ''


