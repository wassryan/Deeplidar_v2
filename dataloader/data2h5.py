# -*- coding: utf-8 -*-
# dataloader: preserve rgb/depth path into list, return list
# .h5: preserve numpy into dict
import os
import os.path
import glob
import numpy as np
import skimage
import skimage.io
import skimage.transform
import torchvision.transforms as transforms
from PIL import Image,ImageFile
import h5py

import torchvision.transforms as transforms
import random
import preprocess

def gtdepth_loader(path):
    # print("........: ", path)
    img = skimage.io.imread(path)
    depth = img *1.0 / 256.0
    # print(img.shape)
    depth = np.reshape(depth, [img.shape[0], img.shape[1], 1]).astype(np.float32)
    return depth

def default_loader(path): # rgb image 
    img = Image.open(path).convert('RGB')
    return img

# def default_loader(path): # rgb image
#     img = skimage.io.imread(path)
#     return img

def input_loader(path): # surface normal(gt depth)
    img = skimage.io.imread(path)
    imgG = skimage.color.rgb2gray(img)
    img = img.astype(np.float32)
    # print("gt: ",img.shape) # (375,1242,3)
    normals = img * 1.0 / 127.5 - np.ones_like(img) * 1.0

    mask = np.zeros_like(img).astype(np.float32)
    mask[:, :, 0] = np.where(imgG > 0, 1.0, 0.0)
    mask[:, :, 1] = np.where(imgG > 0, 1.0, 0.0)
    mask[:, :, 2] = np.where(imgG > 0, 1.0, 0.0)

    return normals,mask

def sparse_loader(path): # lidar depth
    # print("trainLoaderN.py--sparse_loader()")
    img = skimage.io.imread(path)
    img = img * 1.0 / 256.0

    img = img.astype(np.float32) # my add
    mask = np.where(img > 0.0, 1.0, 0.0)
    mask = np.reshape(mask, [img.shape[0], img.shape[1], 1])
    # print("mask: ",mask.shape) # (375,1242,1)
    mask = mask.astype(np.float32)
    img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    # print("img: ", img.shape) # (375,1242,1)
    # print(img.dtype)
    return img,mask

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
def scale_crop2(normalize=__imagenet_stats):
    t_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)

def data2path():
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
        glob_gt1 = os.path.join(root_d,glob_gt) # '/nfs-data/.../workspace/data/kitti_depth' + glob_gt,
        paths_normalgt = sorted(glob.glob(glob_gt1)) #/nfs-data/zhengk_data/normal_data/workspace/kitti_depth/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/*.png
        
        glob_gt2 = os.path.join(root_d2,glob_gt) # '/nfs-data/.../workspace/data/kitti_depth' + glob_gt,
        paths_gt = sorted(glob.glob(glob_gt2)) #/nfs-data/zhengk_data/normal_data/workspace/kitti_depth/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/*.png
        # paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt] # 雷达产生depth的图 train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png
        # 虽然Completion的train中的图是-+5 ID开始的，但是读取rgb图时是以completion中的图片路径替换字符串得到的，所以rgb和gt/depth是对齐的
        # print(paths_gt[0])
        paths_rgb = [get_rgb_paths(p) for p in paths_gt] # /home/zhengk/data/kitti_rgb/train/2011_10_03_drive_0042_sync/image_02/data/*.png
        paths_d = [ get_lidepth_paths(p) for p in paths_rgb ]
        # print(paths_d[0])

    else:
        print("glob_gt is none!!")

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0 and len(paths_normalgt):
        raise(RuntimeError("Found 0 images in data folders"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt) or len(paths_normalgt)!= len(paths_d):
        raise(RuntimeError("Produced different sizes for datasets"))

    return paths_rgb,paths_d,paths_normalgt,paths_gt # rgb image/lidar depth/surface normal(depth gt)/depth gt

'''
if args.model == 'normal':
    all_left_img, all_normal, all_gts = lsn.dataloader(datapath)
    
TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img,all_normal,all_gts ,True, args.model),
        batch_size = args.batch_size, shuffle=True, num_workers=8, drop_last=True)
'''

left_img, lidar_depth, gt_normal, gt_depth = data2path()

# left = default_loader(left_img[0]) # left()
# sparse, mask = sparse_loader(lidar_depth[0]) # sparse(1,256,512) mask(256,512,1)
# normal, mask1 = input_loader(gt_normal[0])# normal(3,256,512) mask1:(256,512,1)
# print(left.dtype) # uint8
# print(left.shape) # (375,1242,3)
# print(type(left))
# print(sparse.dtype, mask.dtype)# (375, 1242, 1), (375, 1242, 1)
# print(type(sparse))
# print(normal.shape, mask1.shape)# (375, 1242, 3), (375, 1242, 3)
# # w,h = left.size # width, height
# num = len(left_img)
# print(num) # 85898

# left_nd = np.zeros((num, left.shape[0], left.shape[1], left.shape[2]))
# sparse_nd = np.zeros((num, sparse.shape[0], sparse.shape[1], sparse.shape[2])).astype(np.float32)
# mask_nd = np.zeros((num, mask.shape[0], mask.shape[1], mask.shape[2])).astype(np.float32)
# normal_nd = np.zeros((num, normal.shape[0], normal.shape[1], normal.shape[2])).astype(np.float32)
# mask1_nd = np.zeros((num, mask1.shape[0], mask1.shape[1], mask1.shape[2])).astype(np.float32)
# print(left_nd.shape)
# print(sparse_nd)

# h5f = h5py.File('./train_normal.h5', 'w')
# print(left_img[0])# /home/zhengk/data/kitti_rgb/train/2011_09_26_drive_0001_sync/image_02/data/0000000005.png
# print(left_img[10000])

train_path = '/nfs-data/zhengk_data/kitti_hdf5/train/'
if not os.path.isdir(train_path):
    os.makedirs(train_path)

print(len(left_img))
print(len(lidar_depth))
print(len(gt_normal))
for idx in range(len(left_img)):
    ps = left_img[idx].split('/')
    pnew = ps[-4] + '_' + ps[-3] + '_' +ps[-1]
    pnew = pnew.split('.')[0]
    # print("+++", idx, pnew)

    print(idx, "**** ", left_img[idx])
    print(idx, "++++ ", lidar_depth[idx])
    print(idx, "---- ", gt_normal[idx])
    print(idx, "#### ", gt_depth[idx])
    h5f = h5py.File(train_path + pnew + '.h5', 'w')
    left = default_loader(left_img[idx])
    sparse, mask = sparse_loader(lidar_depth[idx])
    normal, mask1 = input_loader(gt_normal[idx])
    gtdepth = gtdepth_loader(gt_depth[idx])

    w, h = left.size
    th, tw = 256, 512
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    left = left.crop((x1, y1, x1 + tw, y1 + th))
    normal = normal[y1:y1 + th, x1:x1 + tw, :]
    gtdepth = gtdepth[y1:y1 + th, x1:x1 + tw, :]
    sparse_n = sparse[y1:y1 + th, x1:x1 + tw, :]
    mask = mask[y1:y1 + th, x1:x1 + tw, :]
    mask1 = mask1[y1:y1 + th, x1:x1 + tw, :]

    h5f['rgb'] = left
    h5f['sp'] = sparse_n
    h5f['gtnormal'] = normal
    h5f['gtdepth'] = gtdepth
    h5f['mask'] =  mask # lidar depth mask
    h5f['mask1'] =  mask1 # surface normal mask



