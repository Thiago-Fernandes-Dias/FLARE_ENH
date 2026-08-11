import os
import os.path as osp

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from model.network import VQFPEnhancer_PCNN
from tqdm import tqdm
from utils.misc import load_model

GPU = "0"
FOLDERS = [
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB1_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB1_B"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db2_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB2_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db2_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB2_B"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db3_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB3_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db3_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB3_B"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db4_a/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB4_A"),
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db4_b/bmp", "/mnt/d/Datasets/FVC_FLARE_ENH/priorenh/FVC_2002_DB4_B")
]  # list of (input_folder, output_folder) pairs
W = 0.0
CKPT_PATH = "pretrained_model/priorenh"
METHOD_NAME = "priorenh"
PRE_ENH = False


def image_read(img_path, size=512):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h_org, w_org = img.shape
    if h_org > w_org:
        img = cv2.copyMakeBorder(
            img, 0, 0, 0, h_org - w_org, cv2.BORDER_CONSTANT, value=255
        )
        padding_direction = 0
        padding_size = h_org - w_org
    else:
        img = cv2.copyMakeBorder(
            img, 0, w_org - h_org, 0, 0, cv2.BORDER_CONSTANT, value=255
        )
        padding_direction = 1
        padding_size = w_org - h_org

    img = cv2.resize(img, (size, size))  # try to directly resize the image to 256 x 256
    img = (img.astype(np.float32) - 127.5) / 127.5
    return img[None], (w_org, h_org), padding_direction, padding_size


def inverse_image(image, org_size, padding_direction, padding_size):
    max_size = max(org_size)
    image = cv2.resize(image, (max_size, max_size))
    if padding_direction == 0:
        image = image[:, : max_size - padding_size]
    else:
        image = image[: max_size - padding_size, :]
    return image


def deploy_enh(folders, ckpt_path, w=0, method_name="PriorEnh", pre_enh=False):
    model_path = osp.join(ckpt_path, "priorenh.pth")
    cfg_path = osp.join(ckpt_path, "vq.yaml")
    config = edict(yaml.safe_load(open(cfg_path, "r")))
    model = VQFPEnhancer_PCNN(
        config.hdconfig,
        config.ldconfig,
        n_embed=config.n_codebook,
        embed_dim=config.embed_dim,
        pcn_embed=config.pcn_embed,
        ckpt_path=config.ckpt_path,
        pre_enh=pre_enh,
    )
    model.cuda()
    load_model(model, model_path)
    model.eval()

    for input_folder, output_folder in folders:
        input_folder = os.path.normpath(input_folder)
        img_lst = os.listdir(input_folder)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in tqdm(img_lst, desc=osp.basename(input_folder)):
            img_path = osp.join(input_folder, img_name)
            img, org_shape, padding_dr, padding_size = image_read(img_path)
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            with torch.no_grad():
                try:
                    enh = model.module.enhance(img, w=w)
                except:
                    enh = model.enhance(img, w=w)
            enh = torch.clamp(enh, -1, 1)
            enh = enh.squeeze().cpu().numpy()
            enh = ((enh + 1) * 127.5).astype(np.uint8)
            enh = inverse_image(enh, org_shape, padding_dr, padding_size)
            cv2.imwrite(osp.join(output_folder, img_name), enh)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    deploy_enh(FOLDERS, CKPT_PATH, W, method_name=METHOD_NAME, pre_enh=PRE_ENH)
