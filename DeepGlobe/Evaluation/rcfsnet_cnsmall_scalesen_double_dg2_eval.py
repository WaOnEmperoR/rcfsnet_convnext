import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

# import sklearn.metrics as metrics
import cv2
import os
import numpy as np

from torch.nn import functional as F

from time import time
from PIL import Image

import warnings

warnings.filterwarnings('ignore')

from RCFSNet_ConvNeXt_Small_NoSigmoid_ScaleSen import RCFSNet_CN_Small_NoSigmoid_ScaleSen

source = '../../DG2/test/images'
val = os.listdir(source)

BATCHSIZE_PER_CARD = 3

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
#         print(batchsize)
        
#         if batchsize >= 8:
#             print("enter 8")
#             return self.test_one_img_from_path_1(path)
#         elif batchsize >= 4:
#             print("enter 4")
#             return self.test_one_img_from_path_2(path)
#         elif batchsize >= 2:
#             print("enter 2")
#             return self.test_one_img_from_path_4(path)
        
        return self.test_one_img_from_path_0(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (1024, 1024))
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]
        img1 = img
        img2 = np.array(img1)[::-1]
        img3 = np.array(img1)[:, ::-1]
        img4 = np.array(img2)[:, ::-1]
        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)
        img3 = img3.transpose(2, 0, 1)
        img4 = img4.transpose(2, 0, 1)
        img1 = img1.reshape(-1, 3, 1024, 1024)
        img2 = img2.reshape(-1, 3, 1024, 1024)
        img3 = img3.reshape(-1, 3, 1024, 1024)
        img4 = img4.reshape(-1, 3, 1024, 1024)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask2 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]

        return mask2, maska, maskb, maskc, maskd

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_0(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (1024, 1024))
        img = img.transpose(2, 0, 1)

        img = img.reshape(-1, 3, 1024, 1024)
        
        img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        
        mask = F.sigmoid(self.net.forward(img)).squeeze().cpu().data.numpy()
        
        return mask
        
    def test_one_img_from_path_1(self, path):
#         print("enter path 1")
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        # 修改图片尺寸
        img = cv2.resize(img, (1024, 1024))
#         print(img.shape)
        img = img.transpose(2, 0, 1)
#         print(img.shape)
        img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()

        return mask
    
    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)

solver = TTAFrame(RCFSNet_CN_Small_NoSigmoid_ScaleSen)

solver.load('../Training/weights/RCFSNet_DG2_CnSmall_ScaleSen_double_pos_030_v01_50epoch.th')

target = 'submits/RCFSNet_CnSmall_ScaleSen_DG_double_pos_030_v01/'

tic = time()
if not os.path.exists(target):
    os.makedirs(target)    

gt_root = '../../DG2/test/masks'

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    iou = TP / (TP + FN + FP)
    pre = TP / (TP + FP + 1e-6)
    f1 = (2 * pre * sen) / (pre + sen + 1e-6)
    return acc, sen, iou, pre, f1, TP, FN, TN, FP

import os
import pandas as pd
from tqdm import tqdm

threshold = 2
disc = 20

total_m1 = 0

hausdorff = 0
total_acc = []
total_sen = []
total_iou = []
total_pre = []
total_f1 = []
tp = []
fn= []
tn= []
fp= []

total_auc = []

df = pd.DataFrame(columns=['image_id', 'accuracy', 'iou', 'precision', 'recall', 'f1'])

for i, name in tqdm(enumerate(val)):
    image_path = os.path.join(source, name)
#     print(i)
#     print(image_path)
    
    mask = solver.test_one_img_from_path(image_path)
    
    new_mask = mask.copy()
    
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

    ground_truth_path = os.path.join(gt_root, name.split('.')[0] + '_mask.png')
#     print(ground_truth_path)
    
    ground_truth = np.array(cv2.imread(ground_truth_path))[:, :, 1]
    
    mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

    new_mask = cv2.resize(new_mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

    predi_mask = np.zeros(shape=np.shape(mask))
    predi_mask[mask > disc] = 1
    gt = np.zeros(shape=np.shape(ground_truth))
    gt[ground_truth > 0] = 1
    
    cv2.imwrite(target + name.split('.')[0] + 'mask.png', mask.astype(np.uint8))
    
    acc, sen, iou, pre, f1, TP, FN, TN, FP = accuracy(predi_mask[:, :, 0], gt)
    
#     print("ACC : %3.4f, Sensitivity : %3.4f, Precision : %3.4f" % (acc, sen, pre))   # print integer value
#     print("IoU : %3.4f, F1-Score : %3.4f" % (iou, f1))
#     print("TP : %3.2f, FN : %3.2f, TN : %3.2f, FP : %3.2f" % (TP, FN, TN, FP))
    df.loc[len(df.index)] = [name, acc, iou, pre, sen, f1] 
    
    total_acc.append(acc)
    total_sen.append(sen)
    total_iou.append(iou)
    total_pre.append(pre)
    total_f1.append(f1)
    tp.append(TP)
    fn.append(FN)
    tn.append(TN)
    fp.append(FP)

    # print(i + 1, acc, sen, calculate_auc_test(new_mask / 8., ground_truth))
#     print(i + 1, acc, sen, iou, pre, f1)
        
#     print("--------------")

print("AVERAGE OVER 726 IMAGES")
print(np.mean(total_acc), np.std(total_acc))
print(np.mean(total_sen), np.std(total_sen))
print(np.mean(total_iou), np.std(total_iou))
print(np.mean(total_pre), np.std(total_pre))
print(np.mean(total_f1), np.std(total_f1))
print(np.sum(tp), np.mean(tp))
print(np.sum(fn), np.mean(fn))
print(np.sum(tn), np.mean(tn))
print(np.sum(tp), np.mean(tp))

df.to_csv('rcfsnet_cnsmall_scalesen_test_DG_double_pos_030_v01.csv')