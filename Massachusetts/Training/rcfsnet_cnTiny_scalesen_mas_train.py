import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from networks.RCFSNet_ConvNeXt_Tiny_ScaleSen import RCFSNet_CN_Tiny_ScaleSen
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

from tqdm import tqdm

import Constants2
import image_utils

solver = MyFrame(RCFSNet_CN_Tiny_ScaleSen, dice_bce_loss, 2e-4)
batchsize = 6  

dataset = ImageFolder(root_path='../../massachusetts-roads-dataset/tiff/', datasets='Mas')

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        drop_last =True)

NAME = 'RCFSNet_Mas_CnTiny_ScaleSen_v01_100epoch'
print(NAME)

# start the logging files
mylog = open('logs/' + NAME + '.log', 'w')
tic = time()

no_optim = 0
total_epoch = 100
train_epoch_best_loss = Constants2.INITAL_EPOCH_LOSS

epoch_training_time=[0]
total_time_elapsed=[0]

for epoch in tqdm(range(1, total_epoch + 1)):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    index = 0

    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss, pred = solver.optimize()
        train_epoch_loss += train_loss
        index = index + 1

    # show the original images, predication and ground truth on the visdom.

    # ########归一化的方法为什么？？？
    # show_image = (img + 1.6) / 3.2 * 255.
    # viz.img(name='images', img_=show_image[0, :, :, :])
    # viz.img(name='labels', img_=mask[0, :, :, :])
    # viz.img(name='prediction', img_=pred[0, :, :, :])

    train_epoch_loss = train_epoch_loss/len(data_loader_iter)
    
    time_delta = int(time() - tic)
    
    total_time_elapsed.append(time_delta)
    
    print("previous running time : " + str(epoch_training_time[epoch-1]))
    
    epoch_training_time.append(time_delta - total_time_elapsed[epoch-1])
                     
    # 将信息保存在log文件夹中
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('SHAPE:', Constants2.Image_size, file=mylog)
    print('********')
#     print('epoch:', epoch, '    time:', int(time() - tic))
    print('epoch:', epoch, '    time:', time_delta)
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', Constants2.Image_size)
    

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('./weights/' + NAME + '.th')
    if no_optim > Constants2.NUM_EARLY_STOP:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > Constants2.NUM_UPDATE_LR:
        if solver.old_lr < 5e-7:
            break
        solver.load('./weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print(epoch_training_time)
print(total_time_elapsed)

print('Finish!', file=mylog)
print('Finish!')
mylog.close()