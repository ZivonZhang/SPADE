python train.py  --gpu_ids 1 --name hazeToClear  --continue_train --niter 100

python test.py --name hazeToClear --dataroot [path_to_dataset]


python train.py  --gpu_ids 1 --name ClearToClear  --niter 5 --label_dir /home/tangqf/SSD_disk/Fog_Data/NYU_GT

## 201912228 0920
python train.py  --gpu_ids 1 --name hazeToClear --dataset_mode dhazy --niter 20

## 201912228 1130
python train.py  --gpu_ids 1 --name reside --niter 10 --dataset_mode reside

## 2020-01-03 512 reside
python train.py  --gpu_ids 1 --name reside --niter 10 --dataset_mode reside --continue_train

跑了9 epochs 测试
python test.py --gpu_ids 1 --name reside512 --dataset_mode reside
psnr:  19.784066259676674 ssim: 0.8555852571446189
psnr:  19.920439433431376 ssim: 0.8457911747107557

## mask 以mask来估计呢？  来试试吧 ——mask显然有问题，这个trans是传输图，还需要光照图，才能 生成/还原 有雾或者清晰的图像。

python train.py  --gpu_ids 0 --name mask_reside --niter 10 --dataset_mode reside --mask
python test.py --gpu_ids 0 --name mask_reside --dataset_mode reside

## 2020-01-04 512 d-hazy——512
python train.py  --gpu_ids 1 --name d_hazy_512 --niter 35 --dataset_mode dhazy
python test.py  --gpu_ids 1 --name d_hazy_512  --dataset_mode dhazy

psnr:  16.990694238712404 ssim: 0.8042252164261388

## 2020-01-07 256 d-hazy depth  seg-dep-seg-dep
python train.py  --gpu_ids 1 --name d_hazy_256_depth_opposite --niter 15 --dataset_mode dhazy --use_depth --use_512
python test.py  --gpu_ids 1 --name d_hazy_256_depth1 --dataset_mode dhazy --use_depth -use_512

用了原图了。。。傻了。
psnr:  33.65084299371388 ssim: 0.9601459052381044  卧槽！
其实应该是  psnr:  18.344661964338353 ssim: 0.7978770760069456
psnr:  18.343866410911946 ssim: 0.8070216167726885 用512图进行测


## 2020-01-07 256 reside
python train.py  --gpu_ids 1 --name reside_256 --niter 10 --dataset_mode reside
命令敲错了？虽然名字是reside_256_depth1，其实是reside_256，需注意
python test.py  --gpu_ids 0 --name reside_256 --dataset_mode reside 

psnr:  19.2448085647814 ssim: 0.8215719703997537 这个比256的reside还稍低一点，符合预期


## 2020-01-09 512 reside depth  seg-dep-seg-dep
train.py --gpu_ids 1 --name reside_512_depth1 --niter 5 --dataset_mode reside --use_depth --use_512  --continue_train
python test.py  --gpu_ids 0 --name reside_512_depth1 --dataset_mode reside --use_depth --use_512 True

psnr:  37.30207532703518 ssim: 0.9831989608234246  这个也有点高啊
其实应该是 psnr:  18.56661030114988 ssim: 0.797960463285819


## 2020-01-10 512 reside depth_noCross  dep-dep-seg-seg
python train.py  --gpu_ids 0 --name reside_512_depth_noCross --niter 5 --dataset_mode reside --use_depth --use_512 True --not_use_cross
python test.py  --gpu_ids 1 --name reside_512_depth_noCross --dataset_mode reside --use_depth --use_512 True --not_use_cross


psnr:  18.68968501933805 ssim: 0.7999962935699291


## 2020-01-10 256 dHazy 
python train.py  --gpu_ids 1 --name d_hazy_256 --niter 15 --dataset_mode dhazy --use_512
python test.py  --gpu_ids 1 --name d_hazy_256  --dataset_mode dhazy --use_512

psnr:  16.45366352425129 ssim: 0.7769816935784357

看来提升还是有的

## 2020-01-13 512 D-hazy depth  seg-dep-seg-dep  opposite
python train.py  --gpu_ids 1 --name d_hazy_512_depth_opposite --niter 15 --dataset_mode dhazy --use_depth --use_512
python test.py  --gpu_ids 1 --name d_hazy_512_depth_opposite  --dataset_mode dhazy --use_depth --use_512 1

psnr:  18.170027358071223 ssim: 0.8222026751670172

使用反转的深度图有一定效果

## 2020-01-13 512 D-hazy depth  seg-dep-seg-dep DIF
python train.py  --gpu_ids 0 --name d_hazy_512_depth_DIF --niter 18 --dataset_mode dhazy --use_depth --use_DIF --continue --use_512 0 --notstrict
python test.py  --gpu_ids 0 --name d_hazy_512_depth_DIF --dataset_mode dhazy --use_depth --use_DIF --use_512
psnr:  15.838343457140358 ssim: 0.7694226443134816

使用DIF效果不好

## 2020-01-14 256 D-hazy depth  seg-dep-seg-dep DIF
python train.py  --gpu_ids 1 --name d_hazy_256_depth_DIF1 --niter 15 --dataset_mode dhazy --use_depth --use_DIF --use_512 0 
python test.py  --gpu_ids 1 --name d_hazy_256_depth_DIF1  --dataset_mode dhazy --use_depth --use_DIF --use_512 0 
psnr:  12.257443699125115 ssim: 0.5127268485235905

python train.py  --gpu_ids 1 --name d_hazy_256_depth_opposite --niter 15 --dataset_mode dhazy --use_depth  --dep_opposite --use_512 0 --continue

# Test

## 2020-01-03   256
python test.py --gpu_ids 0 --name reside --dataset_mode reside 
psnr:  18.98716100709391 ssim: 0.8113160307934413



# Loss
GAN: 0.311 GAN_Feat: 2.675 VGG: 2.705 D_Fake: 0.406 D_real: 0.409



# Network Structure












## 2020-01-09 256 d-hazy depth  seg-dep-seg-dep     Fail
```
python train.py  --gpu_ids 1 --name d_hazy_256_depth1 --niter 21 --dataset_mode dhazy --use_depth --continue_train
    python test.py  --gpu_ids 1 --name d_hazy_256_depth1 --dataset_mode dhazy --use_depth

    psnr:  38.60942203478103 ssim: 0.9829272713276112  这有点恐怖的。
```
