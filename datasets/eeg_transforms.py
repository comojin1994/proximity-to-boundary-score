import torch


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor)
