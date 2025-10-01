import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def model_to_DDP(model):
    """Problem 4: model to DDP
    (./handler/DDP/model.py)
    Implement model_to_DDP function.
    model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
    Be careful for set devices. Set profer device id is important part in DDP.
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = model.to(rank % world_size)
    model = DDP(model, device_ids=[rank % world_size])

    return model
