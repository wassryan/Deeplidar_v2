# python test_show_result.py --loadmodel Final_model8.tar
from __future__ import print_function
import argparse
import os
from ipdb import set_trace
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
from submodels import *
from dataloader import preprocess
from PIL import Image
import cv2
import pprint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--loadmodel', default='./trainings/Final_model10.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = s2dN(1)

dummy_left = torch.rand(1, 3, 352, 1216)
dummy_sparse = torch.rand(1, 1, 352, 1216)
dummy_mask = torch.rand(1, 1, 352, 1216)


model = nn.DataParallel(model, device_ids=[0])
model.cuda()

modelpath = os.path.join(ROOT_DIR, args.loadmodel)

if args.loadmodel is not None:
    state_dict = torch.load(modelpath)["state_dict"]
    model.load_state_dict(state_dict)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,sparse,mask):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           sparse = torch.FloatTensor(sparse).cuda()
           mask = torch.FloatTensor(mask).cuda()

        imgL= Variable(imgL)
        sparse = Variable(sparse)
        mask = Variable(mask)

        start_time = time.time()
        with torch.no_grad():
            # with SummaryWriter(comment='Net') as w:
            #     w.add_graph(model, (imgL, sparse, mask))
            # outC, outN, maskC, maskN = model(imgL, sparse, mask)
            outC, outN, maskC, maskN, con_mask = model(imgL, sparse, mask) # zk_add

        tempMask = torch.zeros_like(outC)
        predC = outC[:,0,:,:]
        predN = outN[:,0,:,:]
        tempMask[:, 0, :, :] = maskC
        tempMask[:, 1, :, :] = maskN
        predMask = F.softmax(tempMask)
        predMaskC = predMask[:,0,:,:]
        predMaskN = predMask[:,1,:,:]
        pred1 = predC * predMaskC + predN * predMaskN
        time_temp = (time.time() - start_time)

        output1 = torch.squeeze(pred1)
        # print("***", con_mask.shape)
        confidence_mask = torch.squeeze(con_mask) # zk_add
        # confidence_mask = torch.unsqueeze(confidence_mask, 0)
        # confidence_mask.permute(1,2,0)
        # print("***", confidence_mask.shape)

        # return output1.data.cpu().numpy(),time_temp
        print(output1.shape, confidence_mask.shape)
        return output1.data.cpu().numpy(), time_temp, confidence_mask.data.cpu().numpy() # zk_add

def rmse(gt,img):
    dif = gt[np.logical_and(gt>0, gt<80)] - img[np.logical_and(gt>0, gt<80)]
    error = np.sqrt(np.mean(dif**2))
    return error
def mae(gt,img,ratio):
    dif = gt[np.logical_and(gt>0, gt<80)] - img[np.logical_and(gt>0, gt<80)]
    error = np.mean(np.fabs(dif))
    return error
def absrel(gt,img):
    dif = gt[np.logical_and(gt > 0, gt < 80)] - img[np.logical_and(gt > 0, gt < 80)]
    dif = np.fabs(dif)
    error = np.mean(dif / gt[np.logical_and(gt > 0, gt < 80)])
    return error
def irmse(gt,img):
    dif = 1.0/gt[np.logical_and(gt > 0, gt < 80)] - 1.0/img[np.logical_and(gt > 0, gt < 80)]
    error = np.sqrt(np.mean(dif**2))
    return error
def imae(gt,img):
    dif = 1.0/gt[np.logical_and(gt > 0, gt < 80)] - 1.0/img[np.logical_and(gt > 0, gt < 80)]
    error = np.mean(np.fabs(dif))
    return error

def kitti_disp_to_color(I, max_disp=150):
    assert isinstance(I, np.ndarray), "I should be np.ndarray."
    I = I.astype("float64")
    if max_disp == None:
        max_disp = np.max(I)
    # print(max_disp)
    I = I / max_disp
    h = I.shape[0]
    w = I.shape[1]
    # print("h:",h,"w:",w)
    map = np.array([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
    map = map.astype("float32")
    bins = map[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = np.sum(np.repeat(I.reshape(1, I.size), 6, axis=0) > np.repeat(cbins.reshape(6,1), I.size, axis=1), axis=0)
    ind[ind > 6] = 6
    bins = 1. / bins
    cbins = np.pad(cbins, (1,0), mode="constant")
    I = (I.flatten() - cbins[ind]) * bins[ind]
    I = I.reshape((I.size, 1))
    I_inv = 1 - I
    part1 = map[ind][:, :3] * np.repeat(I_inv, 3, axis=1)
    ind = ind + 1
    part2 = map[ind][:, :3] * np.repeat(I, 3, axis=1)
    parts = part1 + part2
    parts[parts<0] = 0
    parts[parts>1] = 1
    parts = parts * 255
    parts = parts.astype("uint8")
    parts = parts.reshape((h,w,3))
    return parts

def main():
   processed = preprocess.get_transform(augment=False)

   root = '/nfs-data/zhengk_data/kitti_depth/val_selection_cropped'
   gt_fold = root + '/groundtruth_depth/'
   left_fold = root + '/image/'
   lidar2_raw = root + '/velodyne_raw/'

   gt = [img for img in os.listdir(gt_fold)]
   image = [img for img in os.listdir(left_fold)]
   lidar2 = [img for img in os.listdir(lidar2_raw)]
   gt_test = [gt_fold + img for img in gt]
   left_test = [left_fold + img for img in image]
   sparse2_test = [lidar2_raw + img for img in lidar2]
   left_test.sort()
   sparse2_test.sort()
   gt_test.sort()

   time_all = 0.0

   for inx in range(len(left_test)):
       # print(inx)

       imgL_o = skimage.io.imread(left_test[inx])
       # skimage.io.imsave(inp, imgL_o)
       im = cv2.imread(left_test[inx])
       imgL_o = np.reshape(imgL_o, [imgL_o.shape[0], imgL_o.shape[1],3])
       imgL = processed(imgL_o).numpy()
       imgL = np.reshape(imgL, [1, 3, imgL_o.shape[0], imgL_o.shape[1]])

       gtruth = skimage.io.imread(gt_test[inx]).astype(np.float32)
       gtruth = gtruth * 1.0 / 256.0
       sparse = skimage.io.imread(sparse2_test[inx]).astype(np.float32)
       sparse = sparse *1.0 / 256.0


       mask = np.where(sparse > 0.0, 1.0, 0.0)
       mask = np.reshape(mask, [imgL_o.shape[0], imgL_o.shape[1], 1])
       sparse = np.reshape(sparse, [imgL_o.shape[0], imgL_o.shape[1], 1])
       sparse = processed(sparse).numpy()
       sparse = np.reshape(sparse, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])
       mask = processed(mask).numpy()
       mask = np.reshape(mask, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])

       output1 = '/home/zhengk/DeepLiDAR/my_output/' + left_test[inx].split('/')[-1]

       pred, time_temp, confidence_mask = test(imgL, sparse, mask)
       print(type(pred), type(confidence_mask))
       pred = np.where(pred <= 0.0, 0.9, pred)

       time_all = time_all+time_temp
       # print(time_temp)

       
       pred_show = pred * 256.0
       pred_show = pred_show.astype('uint16')
       res_buffer = pred_show.tobytes()
       img = Image.new("I",pred_show.T.shape)
       img.frombytes(res_buffer,'raw',"I;16")
       img.save(output1)
       

       '''
       # show result compare one by one
       print(inx)
       # show color_lidar/confidence mask/pred
       sp = cv2.imread(sparse2_test[inx], -1)
       sp = sp * 1.0/256.0
       
       #pprint.pprint(sp)
       #pprint.pprint(pred)
       #pprint.pprint(confidence_mask)
       #print("sp: ", sp.shape)
       #print("confidence_mask: ", confidence_mask.shape)
       #print("pred: ", pred.shape)

       c_sp = kitti_disp_to_color(sp, max_disp=100)
       c_pred = kitti_disp_to_color(pred, max_disp=100)
       # c_mask = confidence_mask.reshape(confidence_mask.shape[0], confidence_mask.shape[1], 1)
       c_mask = np.expand_dims(confidence_mask, axis=2)
       c_mask = cv2.cvtColor(c_mask, cv2.COLOR_GRAY2RGB)
       
       #pprint.pprint(c_pred)
       #pprint.pprint(c_mask)
       #print("rgb: ", im.shape)
       #print("c_sp: ", c_sp.shape)
       #print("c_mask: ", c_mask.shape)
       #print("c_pred: ", c_pred.shape)
       
       # com = np.hstack((c_sp, c_mask, c_pred))
       # cv2.imshow('dd', com)

       #cv2.imshow("rgb ", im)
       #cv2.imshow('c_sp', c_sp)
       #cv2.imshow('confidence', c_mask)
       #cv2.imshow('c_pred', c_pred)

       #cv2.waitKey(0)
       '''

   print("time: %.8f" % (time_all * 1.0 / 1000.0))

if __name__ == '__main__':
   main()





