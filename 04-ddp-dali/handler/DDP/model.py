import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def model_to_DDP(model):
    ''' Problem 4: model to DDP
    (./handler/DDP/model.py)
    Implement model_to_DDP function.
    model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
    Be careful for set devices. Set profer device id is important part in DDP.
    '''
    raise NotImplementedError()
    return None