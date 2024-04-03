import torch


def verify_cuda():
    if not torch.cuda.is_available():
        print("CUDA NOT available. Running on CPU")
        return False

    print("CUDA IS AVAILABLE. Running on GPU")
    return True
