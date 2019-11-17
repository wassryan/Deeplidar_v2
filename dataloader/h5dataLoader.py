import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py

import preprocess
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.h5', ]

INSTICS = {"2011_09_26": [721.5377, 596.5593, 149.854],
           "2011_09_28": [707.0493, 604.0814, 162.5066],
           "2011_09_29": [718.3351, 600.3891, 159.5122],
           "2011_09_30": [707.0912, 601.8873, 165.1104],
           "2011_10_03": [718.856, 607.1928, 161.2157]
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def h5_loader(path):
    rt = '/nfs-data/zhengk_data/kitti_hdf5/train/'
    h5f = h5py.File(rt + path, "r")
    left = np.array(h5f['rgb']) # convert class 'h5py._hl.dataset.Dataset' into array
    sparse_n = np.array(h5f['sp'])
    #normal = np.array(h5f['gtnormal'])
    gtdepth = np.array(h5f['gtdepth'])
    mask = np.array(h5f['mask']) # lidar depth mask
    #mask1 = np.array(h5f['mask1']) # surface normal mask
    # print(type(left), sparse_n.shape, mask.shape, mask1.shape, normal.shape)
    return left,sparse_n,mask,gtdepth

def get_h5path(path=''):
    return os.listdir(path)


class H5Dataset(data.Dataset):
    def __init__(self, root,loader=h5_loader):
        self.loader = loader
        self.root = root
        self.h5path = get_h5path(root)

    def __getitem__(self, index):
        h5p = self.h5path[index]
        # h5p: /nfs-data/zhengk_data/kitti_hdf5/train/2011_09_26_drive_0009_sync_image_02_0000000002.h5
        left_img, sparse_n, mask, gtdepth = self.loader(h5p) # read h5

        # index_str = self.left[index].split('/')[-4][0:10]
        index_str = h5p.split('/')[-1][0: 10]
        params_t = INSTICS[index_str]
        # print(params_t)
        params = np.ones((256,512,3),dtype=np.float32)
        params[:, :, 0] = params[:,:,0] * params_t[0]
        params[:, :, 1] = params[:, :, 1] * params_t[1]
        params[:, :, 2] = params[:, :, 2] * params_t[2]
       

        params = np.reshape(params, [256, 512, 3]).astype(np.float32)

        # convert array into tensor
        processed = preprocess.get_transform(augment=False) # convert numpy(H, W, C) to FloatTensor(C x H x W)
        # processed = scale_crop2()
        left_img = processed(left_img)
        sparse_n = processed(sparse_n)
        mask = processed(mask)
        # print("left: ", left_img.shape) # (3,256,512)
        # print("------------:", sparse_n.shape) # (1,256,512)
        # print("************:", mask.shape) # (256,512,1)
        return left_img, gtdepth, sparse_n, mask, params
    def __len__(self):
        return len(self.h5path)
