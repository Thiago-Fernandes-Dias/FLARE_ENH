# -*- encoding: utf-8 -*-
'''
@File    :   deploy_enh.py
@Time    :   2024/10/09 16:20:23
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2024, Zhiyu Pan, Tsinghua University. All rights reserved
@Function :  deploy the model to a specific dataset
'''
import torch
import os.path as osp
import os
import cv2
from model.network import SqueezeUNet
from utils.misc import load_model
import numpy as np
from tqdm import tqdm

GPU = '0'
FOLDERS = [
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB1_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB1_B"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db2_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB2_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db2_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB2_B"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db3_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB3_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db3_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB3_B"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db4_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB4_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db4_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/unetenh/FVC_2002_DB4_B")
]  # list of (input_folder, output_folder) pairsCKPT_PATH = 'pretrained_model/unetenh/unetenh.pth'
METHOD_NAME = 'UNetEnh'
PRE_ENH = False
MODEL_PATH = "pretrained_model/unetenh/unetenh.pth"

def image_read(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # process the img size for 4 
    h_org, w_org = img.shape
    # find the nearest 4 times of the img size, for the larger direction
    h = int(np.ceil(h_org / 16) * 16)
    w = int(np.ceil(w_org / 16) * 16)
    img = cv2.resize(img, (w, h)) 
    img = (img.astype(np.float32) - 127.5) / 127.5
    return img[None], (w_org, h_org)


def deploy_enh(folders, model_path, method_name, pre_enh=False):
    model = SqueezeUNet(input_channels=1, num_classes=2, pre_enh=pre_enh)
    model.cuda()
    load_model(model, model_path)
    model.eval()

    for input_folder, output_folder in folders:
        input_folder = os.path.normpath(input_folder)
        img_lst = os.listdir(input_folder)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in tqdm(img_lst, desc=osp.basename(input_folder)):
            img_path = osp.join(input_folder, img_name)
            img, org_shape = image_read(img_path)
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            with torch.no_grad():
                pred = model(img)
            enh, _ = torch.split(pred, [1, 1], dim=1)
            enh = enh.squeeze().cpu().numpy()
            enh = (enh * 255).astype(np.uint8)
            enh = cv2.resize(enh, org_shape)
            cv2.imwrite(osp.join(output_folder, img_name), enh)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    deploy_enh(FOLDERS, MODEL_PATH, METHOD_NAME, pre_enh=PRE_ENH)