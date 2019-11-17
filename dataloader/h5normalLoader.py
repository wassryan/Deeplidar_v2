import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py

import preprocess
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.h5', ]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def h5_loader(path):
    rt = '/nfs-data/zhengk_data/kitti_hdf5/train/'
    h5f = h5py.File(rt + path, "r")
    left = np.array(h5f['rgb']) # convert class 'h5py._hl.dataset.Dataset' into array
    sparse_n = np.array(h5f['sp'])
    normal = np.array(h5f['gtnormal'])
    mask = np.array(h5f['mask']) # lidar depth mask
    mask1 = np.array(h5f['mask1']) # surface normal mask
    # print(type(left), sparse_n.shape, mask.shape, mask1.shape, normal.shape)
    return left,sparse_n,mask,mask1,normal

def get_h5path(path=''):
    return os.listdir(path)


class H5Dataset(data.Dataset):
    def __init__(self, root,loader=h5_loader):
        self.loader = loader
        self.root = root
        self.h5path = get_h5path(root)

    def __getitem__(self, index):
        h5p = self.h5path[index]
        left_img, sparse_n, mask, mask1, normal = self.loader(h5p) # read h5

        # convert array into tensor
        processed = preprocess.get_transform(augment=False) # convert numpy(H, W, C) to FloatTensor(C x H x W)
        # processed = scale_crop2()
        left_img = processed(left_img)
        sparse_n = processed(sparse_n)
        # print("left: ", left_img.shape) # (3,256,512)
        # print("------------:", sparse_n.shape) # (1,256,512)
        # print("************:", mask.shape) # (256,512,1)
        return left_img, sparse_n, mask, mask1, normal
    def __len__(self):
        return len(self.h5path)
