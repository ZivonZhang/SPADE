"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256 #286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)  #13
        parser.set_defaults(contain_dontcare_label=False)
        '''
            parser.add_argument('--label_dir', type=str, default = '/home/tangqf/SSD_disk/Fog_Data/NYU_Hazy',
                                help='path to the directory that contains label images')
            parser.add_argument('--image_dir', type=str, default = '/home/tangqf/SSD_disk/Fog_Data/NYU_GT',
                                help='path to the directory that contains photo images')
            parser.add_argument('--instance_dir', type=str, default='',
                                help='path to the directory that contains instance maps. Leave black if not exists')
        '''
        # Resume
        parser.add_argument('--data_root', type=str, default='/home/tangqf/SSD_disk/ITS/ITS')
        parser.add_argument('--train_file',type=str, default='train_file.txt')
        parser.add_argument('--val_file',type=str, default='val_file.txt')

        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        instance_paths = []

        return label_paths, image_paths, instance_paths  # 返回绝对路径

    def __getitem__(self, index):
        if 'ITS' in self.args.data_root or 'OTS' in self.args.data_root:
            haze_path_temp, gt_path = self.img_list[index].split(' ')    # `train/ITS_haze/9542_* train/ITS_clear/9542.png`
            haze_path = random.choice(glob.glob(os.path.join(self.args.data_root, haze_path_temp)))
            gt_path = os.path.join(self.args.data_root, gt_path)
            if self.args.depth or self.args.depth_out:
                depth_path = haze_path.replace(self.args.data_root,self.args.depth_root,1)  # 相应的深度图路径
                gt_dp_path = gt_path.replace(self.args.data_root,self.args.depth_root,1)
        else:
            haze_path, gt_path = self.img_list[index].split(' ')


        # Label Image 
        label_path = self.label_paths[index]

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
        image_path = self.image_paths[index]
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

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

