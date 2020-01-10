import skimage
import cv2
import os
import glob
from multiprocessing import Pool
import time
import sys

time3 = time.time()

name = str(sys.argv[1])
basedir = os.path.join(r'TESTresults', name,r'test_latest/images')

dh_files = glob.glob(basedir+r'/synthesized_image/*.png') # TESTresults\reside\test_latest\images\synthesized_image\*.png
gt_files = glob.glob(basedir+r'/ground_truth/*.png') # ground_truth
dh_files.sort()
gt_files.sort()
num = len(dh_files)

psnr_list = []
ssim_list = []
results = []

multiProcessing = True

def subTask(start:int,end:int)->list:  # 对全局变量进行处理
    tmp_psnr = []
    tmp_ssim = []
    print('Subprocesses %d - %d begin...' %(start,end))
    time1 = time.time()
    for i in range(start,end):
        dh_img = cv2.imread(dh_files[i],0)
        gt_img = cv2.imread(gt_files[i],0)
        psnr = skimage.measure.compare_psnr(dh_img, gt_img, 255)
        ssim = skimage.measure.compare_ssim(dh_img, gt_img, data_range=255)

        tmp_psnr.append(psnr)
        tmp_ssim.append(ssim)

    time2 = time.time()
    print('Subprocesses %d - %d done... Task runs %0.2f seconds.' %(start,end , time2-time1))
    return tmp_psnr,tmp_ssim 

    #return psnr_list,ssim_list


if multiProcessing:
    cpus = 6
    p = Pool(cpus)
    singlenum = num//cpus +1
    for i in range(cpus-1):
        res = p.apply_async(subTask, args=(i *singlenum ,(i+1)*singlenum))
        #print(type(res))
        results.append(res)

    res =p.apply_async(subTask, args=((cpus-1) *singlenum ,num))
    results.append(res)

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

else:
    for i in range(num):
        dh_img = cv2.imread(dh_files[i],0)
        gt_img = cv2.imread(gt_files[i],0)

        psnr = skimage.measure.compare_psnr(dh_img, gt_img, 255)
        ssim = skimage.measure.compare_ssim(dh_img, gt_img, data_range=255)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    
for res in results:
    psnr_list+=res.get()[0]
    ssim_list+=res.get()[1]
    
    
print('psnr: ', sum(psnr_list)/num, 'ssim:', sum(ssim_list)/num)

time4 = time.time()
print('All subprocesses done. Task runs %0.2f seconds.' % (time4-time3))