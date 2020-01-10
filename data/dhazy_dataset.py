"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class DhazyDataset(Pix2pixDataset):
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

        parser.add_argument('--label_dir', type=str, default = '/home/tangqf/SSD_disk/Fog_Data/NYU_Hazy',
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default = '/home/tangqf/SSD_disk/Fog_Data/NYU_GT',
                            help='path to the directory that contains photo images')
        parser.add_argument('--depth_dir', type=str, default = '/home/Data/ZivonZhang/bts/D_hazy',
                            help='path to the directory that contains depth images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if opt.use_depth:
            depth_dir = opt.depth_dir
            depth_paths = make_dataset(depth_dir, recursive=False, read_cache=True)
            assert len(label_paths) == len(depth_paths), "The #images in %s and %s do not match. Is there something wrong?"
        else:
            depth_paths = []

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        splitPot = len(label_paths)//10*9
        #print(splitPot)
        
        label_paths.sort()
        image_paths.sort()
        depth_paths.sort()

        

        #for path1, path2 in zip(label_paths, image_paths):
        #    print(path1,path2)

        if  opt.isTrain:
            return label_paths[:splitPot], image_paths[:splitPot], instance_paths[:len(instance_paths)//10*9], depth_paths[:len(depth_paths)//10*9]
        else:  # Test
            return label_paths[splitPot:], image_paths[splitPot:], instance_paths[len(instance_paths)//10*9:], depth_paths[len(depth_paths)//10*9:]
