import numpy as np
import glob
import cv2
from ipdb import set_trace
import os

folder = './my_output'
img_name_list = glob.glob(folder + '/*.png')
print(len(img_name_list))

c_folder = './my_c_output'
c_img_name_list = glob.glob(folder + '/*.png')
'''
begin = 300
width2 = 704 - begin
writer = cv2.VideoWriter("./mono.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 4, (2000, 2000), True)


fps = 24 
size = (352, 1216)
writer = cv2.VideoWriter(c_folder +"mono.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
'''
'''
def img2video(path = './c_output/'):
    fps = 10
    size = (1280, 720)
    videowriter = cv2.VideoWriter("./mono.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    for i,imgname in enumerate( os.listdir(path)):
        img = cv2.imread(path + imgname)
        print(path+imgname)
        print(img.shape)
        videowriter.write(img)
'''
def img2video(path = './c_output/'):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('./test.avi', fourcc, fps, (1080,1920))
    for i,imgname in enumerate(os.listdir(path)):
        img12 = cv2.imread(path + imgname)
        # cv2.imshow('img', img12)
        # cv2.waitKey(1000/int(fps))
        print(i, img12.shape)
        videoWriter.write(img12)
    videoWriter.release()
img_name_list.sort()

sum_error = 0
count = 0

def save_video_signel(img_name_list, writer):
    for index, line in enumerate(img_name_list):
        ke = cv2.imread(line)
        print(index,"----", ke.shape)
        writer.write(ke)

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


# c_img_name_list = [ p.replace("output", "c_output") for p in img_name_list]

for i,imgname in enumerate(img_name_list):
    c_imgname = imgname.replace("my_output", "my_c_output")
    pred_disp = cv2.imread(imgname, -1)
    pred_disp = pred_disp * 1.0 / 256.0
    print(i,pred_disp.shape)
    res = kitti_disp_to_color(pred_disp,  max_disp=100)
    cv2.imwrite(c_imgname, res) 
'''
# single image
imgname = "./2011_10_03_drive_0047_sync_velodyne_raw_0000000782_image_03.png"
pred_disp = cv2.imread(imgname, -1)
pred_disp = pred_disp * 1.0 / 256.0
print(pred_disp.shape)
res = kitti_disp_to_color(pred_disp,  max_disp=100)
cv2.imwrite("./color_2011_10_03_drive_0047_sync_velodyne_raw_0000000782_image_03.png", res) 
'''
# save_video_signel(c_img_name_list, writer)
# img2video()
