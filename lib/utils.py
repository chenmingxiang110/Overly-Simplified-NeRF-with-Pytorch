import os
import cv2
import torch
import random
import numpy as np

def isfloat(x):
    try:
        float(x)
    except ValueError:
        return False
    return True

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return None
