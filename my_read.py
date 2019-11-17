import cv2

img=cv2.imread('/home/zhengk/data/kitti_depth/train/2011_10_03_drive_0042_sync/proj_depth/groundtruth/image_02/0000000586.png',-1)
#img=cv2.imread('/nfs-data/zhengk_data/normal_data/workspace/kitti_depth/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png')
#print(img)
print(img.shape)
