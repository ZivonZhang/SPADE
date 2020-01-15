"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os
import glob
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util


class ResideDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        
        if parser.get_default('use_512'):
            load_size = 576 if is_train else 512 #286 if is_train else 256
            parser.set_defaults(load_size=load_size)
            parser.set_defaults(crop_size=512)
            parser.set_defaults(display_winsize=512)
        else:
            load_size = 286 if is_train else 256 #286 if is_train else 256
            parser.set_defaults(load_size=load_size)
            parser.set_defaults(crop_size=256)
            parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)  #13
        parser.set_defaults(contain_dontcare_label=False)
        # Resume
        parser.add_argument('--data_root', type=str, default = r'/home/tangqf/SSD_disk/ITS/ITS')
        parser.add_argument('--depth_dir', type=str, default = r'/home/Data/ZivonZhang/bts/Pro_fog',
                            help='path to the directory that contains depth images')
        parser.add_argument('--train_file',type=str, default=r'train_file.txt')
        parser.add_argument('--val_file',type=str, default=r'val_file.txt')

        return parser
        
    def initialize(self, opt):
        self.opt = opt
        ## 
        if opt.isTrain:
            data_file = os.path.join(opt.data_root, opt.train_file)
        else:
            data_file = os.path.join(opt.data_root, opt.val_file)

        print("Load data_file:" , data_file)
        
        with open(data_file, 'r') as f:  # 此处读入txt内存储的相对地址
            data = f.readlines() 
        self.img_list = [item.strip() for item in data]

        size = len(self.img_list)
        self.dataset_size = size

    def number_match(self, path1, path2):
        #if not self.opt.mask:
        filename1_without_ext = os.path.basename(path1).split('_')[0]
        filename2_without_ext = os.path.basename(path2).split('.')[0]
        #else:
        #    filename1_without_ext = os.path.basename(path1).split('_')[0]
         #   filename2_without_ext = os.path.basename(path2).split('_')[0]

        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        if 'ITS' in self.opt.data_root or 'OTS' in self.opt.data_root:
            #if not self.opt.mask:
            haze_path_temp, gt_path = self.img_list[index].split(' ')    # `train/ITS_haze/9542_* train/ITS_clear/9542.png`
            haze_path = random.choice(glob.glob(os.path.join(self.opt.data_root, haze_path_temp)))
            gt_path = os.path.join(self.opt.data_root, gt_path)
            if self.opt.use_depth :
                #depth_path = haze_path.replace(self.args.data_root,self.args.depth_root,1)  # 相应的深度图路径
                gt_dp_path = gt_path.replace(self.opt.data_root,self.opt.depth_dir,1)
            #else:
            #    haze_path_temp, gt_path = self.img_list[index].split(' ')    # `train/ITS_haze/9542_* train/ITS_clear/9542.png`
            #    haze_path = random.choice(glob.glob(os.path.join(self.opt.data_root, haze_path_temp)))

            #    gt_path = gt_path.replace('clear','trans').replace('.png','_'+os.path.basename(haze_path).split('_')[1] + '.png')  # train/ITS_trans/9542_01.png`
            #    gt_path = os.path.join(self.opt.data_root, gt_path)
        else:
            raise ValueError("Not Reside Dataset")

        # Label Image 
        label_path = haze_path

        label = Image.open(label_path)
        label = label.convert('RGB')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params) #method=Image.NEAREST , normalize=False
        #transform_label = get_transform(self.opt, params)
        label_tensor = transform_label(label)
        #print(label_tensor)
        
        # pre
        #label_tensor = transform_label(label) * 255.0
        #label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images) target
        image_path = gt_path
        assert self.number_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

         # if using depth information
        if not self.opt.use_depth:
            depth_tensor = 0
        else:
            depth_path = gt_dp_path
            #print(depth_path)
            depth = Image.open(depth_path)
            depth = depth.convert('RGB')

            transform_depth = get_transform(self.opt, params, isdepth=True)
            depth_tensor = transform_depth(depth)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'depth':depth_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

