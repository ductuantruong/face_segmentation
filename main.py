#from parameter import *
from trainer import Trainer
from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
from argparse import ArgumentParser

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True
    if config.train:
    # Create directories if not exist
        make_folder(config.model_save_path, config.version)
        # make_folder(config.sample_path, config.version)
        make_folder(config.log_path, config.version)

        train_loader = Data_Loader(config.img_path, config.label_path, config.imsize,
                             config.batch_size, config.train)
        eval_loader = Data_Loader(config.val_img_path, config.val_label_path, config.imsize,
                             config.batch_size, False)
        trainer = Trainer(train_loader.loader(), eval_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(config)
        tester.test()

if __name__ == '__main__':
    #config = get_parameters()
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--sample_path', type=str, default=False)
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--total_epoch', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=int, default=None)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--img_path', type=str, default='data/CelebAMaskHQ/train/train_image')
    parser.add_argument('--label_path', type=str, default='data/CelebAMaskHQ/train/train_mask') 
    parser.add_argument('--val_img_path', type=str, default='data/CelebAMaskHQ/val/val_image') 
    parser.add_argument('--val_label_path', type=str, default='data/CelebAMaskHQ/val/val_mask') 
    parser.add_argument('--test_img_path', type=str, default='data/CelebAMaskHQ/test/test_image') 
    parser.add_argument('--test_result_path', type=str, default='./test_results') 
    
    parser.add_argument('--log_epoch', type=int, default=10)

    config = parser.parse_args()

    # print(config)
    main(config)
