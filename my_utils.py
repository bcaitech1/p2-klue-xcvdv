import random
import torch
import os
import numpy as np


def set_seed(seed: object = 42) -> object:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device() -> object:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"set device : {device}")
    return device
