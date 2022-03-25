
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

from fpn.FPN import FPN
from fpn.resnet import resnet

from utils import *
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/training')

class Trainer(object):
    def __init__(self, data_loader, eval_loader, config):

        # Data loader
        self.train_loader = data_loader
        self.eval_loader = eval_loader
        # exact model and loss
        self.model = config.model

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
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if not self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        min_epoch_loss = float('inf')
        training_train_loss = []
        training_eval_loss = []
        for epoch in range(start, self.total_epoch):
            self.G.train()
            epoch_train_loss = []
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for imgs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
                    labels_real_plain = labels[:, 0, :, :].to(device)
                    imgs = imgs.to(device)
                    labels_predict = self.G(imgs)
                    c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                    epoch_train_loss.append(c_loss.data.item())
                    tepoch.set_postfix(loss=c_loss.data)
                    self.reset_grad()
                    c_loss.backward()
                    self.g_optimizer.step()
            avg_train_loss = round(sum(epoch_train_loss)/len(epoch_train_loss), 6)
            training_train_loss.append(avg_train_loss)
            with open('fpn_log.txt', 'a') as f:
                f.write("Epoch {}: Avg Train Loss: {}".format(epoch, avg_train_loss))
                f.write('\n')
            epoch_val_loss = []
            with tqdm(self.eval_loader, unit="batch") as eepoch:
                for imgs, labels in eepoch:
                    eepoch.set_description(f"Evaluating Epoch {epoch}")
                    labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
                    labels_real_plain = labels[:, 0, :, :].to(device)
                    imgs = imgs.to(device)
                    labels_predict = self.G(imgs)
                    c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                    eepoch.set_postfix(loss=c_loss.data)
                    self.reset_grad()
                    c_loss.backward()
                    self.g_optimizer.step()
                    epoch_val_loss.append(c_loss.data.item())
            avg_val_loss = round(sum(epoch_val_loss)/len(epoch_val_loss), 6)
            training_eval_loss.append(avg_val_loss)
            with open('fpn_log.txt', 'a') as f:
                f.write("Epoch {}: Avg Eval Loss: {}".format(epoch, avg_val_loss))
                f.write('\n')
            writer.add_scalar('Loss/Cross_entrophy_loss', avg_val_loss, epoch) 
            if epoch % 1 == 0:
                # min_epoch_loss = avg_epoch_loss
                # print('Saving new best model... Loss: {}'.format(min_epoch_loss))
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(epoch + 1)))
        print('training_train_loss: ', training_train_loss)
        print('training_val_loss: ', training_eval_loss)
    def build_model(self):
        blocks = [2,4,23,3]
        self.G = FPN(blocks, num_classes=19, back_bone='resnet101').to(device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])

        # print networks
        print(self.G)

    def build_tensorboard(self):
        pass #from logger import Logger
        #self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
