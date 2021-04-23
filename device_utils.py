import torch

def current_device(use_gpu: bool):
    if(use_gpu and torch.cuda.is_available()):
        print("using gpu ...\n")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        if use_gpu:
            print("requesting gpu but unavailable, switching to cpu ...\n")
        else:
            print("using cpu ...\n")
        return torch.device("cpu")