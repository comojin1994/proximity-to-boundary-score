def torch2np(x_torch):
    x_np = x_torch.detach().cpu().numpy()
    return x_np
