import os
import shutil
import argparse
from datetime import datetime

import torch
from torch.backends import cudnn
from solver import Solver
from data_loader import get_loader, get_loader_class

def str2bool(v):
    return v.lower() in ('true')

def clean_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main(config):
    cudnn.benchmark = True

    if config.mode == 'train':
        # Checks if the log folder exists.
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        # Checks if the model folder exists and clean it.
        clean_and_create_dir(config.model_save_dir)

        # Checks if the sample folder exists and clean it.
        clean_and_create_dir(config.sample_dir)

        # Checks if the result folder exists and clean it.
        clean_and_create_dir(config.result_dir)

    if config.mode == 'test':
        # Checks if the result folder exists and clean it.
        clean_and_create_dir(config.result_dir)
        
    imagenet_loader = None
    imagenet_class_loader = None

    if config.dataset == 'ImageNet':
        imagenet_loader = get_loader(config.imagenet_image_dir, config.attr_path, config.selected_attrs, config.crop_size, config.image_size,
                                     config.batch_size, 'ImageNet', config.mode, config.num_workers)

        imagenet_class_loader = get_loader_class(config.imagenet_image_dir, config.attr_path, config.selected_attrs, config.crop_size, config.
                                                 image_size, config.batch_size, 'ImageNet', config.mode, config.num_workers)
    
    solver = Solver(imagenet_loader, imagenet_class_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'AFHQ', 'ImageNet']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'AFHQ', 'ImageNet']:
            solver.test()
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=40)
    parser.add_argument('--crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=16)
    parser.add_argument('--d_conv_dim', type=int, default=16)
    parser.add_argument('--c_conv_dim', type=int, default=16) 
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--c_repeat_num', type=int, default=6)     
    parser.add_argument('--lambda_cls', type=float, default=0.25)  
    parser.add_argument('--lambda_rec', type=float, default=1.3)
    parser.add_argument('--lambda_gp', type=float, default=1)
    parser.add_argument('--lambda_perturbation', type=float, default=0.1)
    parser.add_argument('--nadir_slack', type=float, default=1.05)          # Can range between 1.1 and 1.05.
                                            
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_iters', type=int, default=500000)
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

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=1000000)

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--attr_path', type=str, default='../../Datasets/ImageNet5/image_data.txt')
    parser.add_argument('--imagenet_image_dir', type=str, default='../../Datasets/ImageNet5/Images')
    parser.add_argument('--log_dir', type=str, default='logs/imagenet')
    parser.add_argument('--model_save_dir', type=str, default='models/imagenet')
    parser.add_argument('--sample_dir', type=str, default='samples/imagenet')
    parser.add_argument('--result_dir', type=str, default='results/imagenet')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()

    # Writing the parsing arguments into a logging file.
    log_file = os.path.join(config.log_dir, 'log_arguments.txt')
    with open(log_file, mode="a") as log_arg:
        log_arg.write(f"Training session created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")
        log_arg.write(f"{str(config)}\n")
        log_arg.write(f"---\n")
    print("[ INFO ] Parameters saved into '{}'.".format(log_file))
    
    main(config)