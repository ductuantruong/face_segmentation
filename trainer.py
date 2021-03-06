
import os
import time
import torch
import datetime
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

from unet import unet
from deeplab.deeplabv3 import DeepLabV3
from utils import *
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/training')

class Trainer(object):
    def __init__(self, train_loader, eval_loader, config):

        # Data loader
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # exact model and loss
        # self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_epoch = config.total_epoch
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.use_tensorboard = config.use_tensorboard
        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_epoch = config.log_epoch
        # self.sample_epoch = config.sample_epoch
        # self.model_save_epoch = config.model_save_epoch
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        # self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if not self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator
        # data_iter = iter(self.train_loader)
        # step_per_epoch = len(self.train_loader)

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        # start_time = time.time()
        min_epoch_loss = float('inf')
        for epoch in range(start, self.total_epoch):
            self.G.train()
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for imgs, labels in tepoch:
                # try:
                    # imgs, labels = next(data_iter)
                # except:
                    # data_iter = iter(self.train_loader)
                    # imgs, labels = next(data_iter)
                    tepoch.set_description(f"Epoch {epoch}")
                    size = labels.size()
                    labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
                    labels_real_plain = labels[:, 0, :, :].to(device)
                    # labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
                    # oneHot_size = (size[0], 19, size[2], size[3])
                    # labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                    # labels_real = labels_real.scatter_(1, labels.data.long().to(device), 1.0)

                    imgs = imgs.to(device)
                    # ================== Train G =================== #
                    labels_predict = self.G(imgs)
                            
                    # Calculate cross entropy loss
                    c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                    tepoch.set_postfix(loss=c_loss.data)
                    self.reset_grad()
                    c_loss.backward()
                    self.g_optimizer.step()
            
            self.G.eval()
            epoch_loss = []
            with tqdm(self.eval_loader, unit="batch") as eepoch:
                for imgs, labels in eepoch:
                # try:
                    # imgs, labels = next(data_iter)
                # except:
                    # data_iter = iter(self.train_loader)
                    # imgs, labels = next(data_iter)
                    eepoch.set_description(f"Evaluating Epoch {epoch}")
                    size = labels.size()
                    labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
                    labels_real_plain = labels[:, 0, :, :].to(device)
                    # labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
                    # oneHot_size = (size[0], 19, size[2], size[3])
                    # labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                    # labels_real = labels_real.scatter_(1, labels.data.long().to(device), 1.0)

                    imgs = imgs.to(device)
                    # ================== Evaluate G =================== #
                    labels_predict = self.G(imgs)
                            
                    # Calculate cross entropy loss
                    c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                    eepoch.set_postfix(loss=c_loss.data)
                    epoch_loss.append(c_loss.data.item())
            avg_epoch_loss = round(sum(epoch_loss) / len(epoch_loss), 4)
            # Print out log info
            # if (epoch + 1) % self.log_epoch == 0:
                # elapsed = time.time() - start_time
                # elapsed = str(datetime.timedelta(seconds=elapsed))
                # print("Elapsed [{}], G_epoch [{}/{}], Cross_entrophy_loss: {:.4f}".
                    #   format(elapsed, epoch + 1, self.total_epoch, epoch_loss))

            # label_batch_predict = generate_label(labels_predict, self.imsize)
            # label_batch_real = generate_label(labels_real, self.imsize)

            # scalr info on tensorboardX
            writer.add_scalar('Loss/Cross_entrophy_loss', avg_epoch_loss, epoch) 

            # image infor on tensorboardX
            # img_combine = imgs[0]
            # real_combine = label_batch_real[0]
            # predict_combine = label_batch_predict[0]
            # for i in range(1, self.batch_size):
                # img_combine = torch.cat([img_combine, imgs[i]], 2)
                # real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                # predict_combine = torch.cat([predict_combine, label_batch_predict[i]], 2)
            # writer.add_image('imresult/img', (img_combine.data + 1) / 2.0, epoch)
            # writer.add_image('imresult/real', real_combine, epoch)
            # writer.add_image('imresult/predict', predict_combine, epoch)

            # Sample images
            # if (epoch + 1) % self.sample_epoch == 0:
                # labels_sample = self.G(imgs)
                # labels_sample = generate_label(labels_sample)
                # labels_sample = torch.from_numpy(labels_sample)
                # save_image(denorm(labels_sample.data),
                        #    os.path.join(self.sample_path, '{}_predict.png'.format(epoch + 1)))
            if avg_epoch_loss < min_epoch_loss:
                min_epoch_loss = avg_epoch_loss
                print('Saving new best model... Loss: {}'.format(min_epoch_loss))
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(epoch + 1)))
    
    def build_model(self):

        # self.G = unet().to(device)
        self.G = DeepLabV3().to(device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])

        # print networks
        print(self.G)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (epoch: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
