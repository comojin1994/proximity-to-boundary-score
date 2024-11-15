from easydict import EasyDict
from datetime import datetime
import os
import yaml
import torch
import lightning as L


def get_configs():
    config_path = "configs/"
    with open(os.path.join(config_path, "base.yaml"), "r") as file:
        base_config = yaml.safe_load(file)
        args = EasyDict(base_config)

    with open(os.path.join(config_path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
        args.update(config)

    with open(os.path.join(config_path, f"datasets/{args.dataset}.yaml"), "r") as file:
        dataset_config = yaml.safe_load(file)
        args.update(dataset_config)

    with open(os.path.join(config_path, f"models/{args.model}.yaml"), "r") as file:
        model_config = yaml.safe_load(file)
        args.update(model_config[args.dataset])

    return args


def init_configs(args):
    args.current_time = datetime.now().strftime("%Y%m%d")

    # Set Device
    args.device = get_device(args.GPU_NUM)

    return args


def get_device(GPU_NUM: str) -> torch.device:
    if torch.cuda.device_count() == 1:
        output = torch.device(f"cuda:{GPU_NUM}")
    elif torch.cuda.device_count() > 1:
        output = torch.device(f"cuda")
    else:
        output = torch.device("cpu")

    print(f"{output} is checked")
    return output


def init_settings(args):

    L.seed_everything(args.SEED)
