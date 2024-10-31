"""
Created on Wed Oct 30 15:17:43 2024

@author: joseareia
"""

import os
import torch
import argparse
from solver import Solver
from data_loader import get_loader, get_loader_class


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    torch.backends.cudnn.benchmark = True
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
        
    celeba_loader = None
    celeba_class_loader = None
    afhq_loader = None
    afhq_class_loader = None
    imagenet_loader = None
    imagenet_class_loader = None       
    imagenet_loader = None
    imagenet_class_loader = None

    imagenet_loader = get_loader(config.imagenet_image_dir, None, 
                                 config.attr_path, config.selected_attrs, 
                                 config.crop_size, config.image_size,
                                 config.batch_size, 'ImageNet', config.mode,
                                 config.num_workers)
    
    imagenet_class_loader = get_loader_class(config.imagenet_image_dir,
                                             config.attr_path, 
                                             config.selected_attrs,
                                             config.crop_size,
                                             config.image_size, 
                                             config.batch_size, 'ImageNet',
                                             config.mode, config.num_workers)
    

    solver = Solver(celeba_loader, celeba_class_loader, afhq_loader, 
                    afhq_class_loader, imagenet_loader, imagenet_class_loader, 
                    config)

    solver.test()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--c_dim', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=8)
    parser.add_argument('--d_conv_dim', type=int, default=8)
    parser.add_argument('--c_conv_dim', type=int, default=8) 
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--c_repeat_num', type=int, default=6)     
    parser.add_argument('--lambda_cls', type=float, default=0.25)  
    parser.add_argument('--lambda_rec', type=float, default=1.3)
    parser.add_argument('--lambda_gp', type=float, default=1)
                        
    # It can be: CelebA, AFHQ or ImageNet.                    
    parser.add_argument('--dataset', type=str, default='ImageNet')
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=1000000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.00012)      
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--c_beta1', type=float, default=0.9)
    
    parser.add_argument('--resume_iters', type=int, default=None)    
    parser.add_argument('--selected_attrs', '--list', nargs='+', default=['original', 'perturbation'])

    parser.add_argument('--test_iters', type=int, default=1000000)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    parser.add_argument('--attr_path', type=str, default='../Datasets/ImageNet5/image_data.txt')
    parser.add_argument('--imagenet_image_dir', type=str, default='../Datasets/ImageNet5/Images')
    parser.add_argument('--log_dir', type=str, default='superstarGAN/logs')
    parser.add_argument('--model_save_dir', type=str, default='superstarGAN/models')
    parser.add_argument('--sample_dir', type=str, default='superstarGAN/samples')
    parser.add_argument('--result_dir', type=str, default='superstarGAN/results')
    
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()  
    main(config)