import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# basic rule of func:
#   1st argument: process index
#   2nd argument: collection of args, including addr, port and num_gpu
def run_process(func, args):
    """Problem 1: Run process
    (./handler/DDP/utils.py)
    Implement run_process function.
    run_process function is used to run main_func in multiple processes, for DDP GPU group.
    It is a wrapper of mp.spawn function.
    You can use mp.spawn function as a reference.
    """
    nprocs = args.num_gpu
    mp.spawn(  # type: ignore
        fn=func,
        args=args,
        nprocs=nprocs,
    )


def initialize_group(proc_id, host, port, num_gpu):
    """Problem 2: Setup GPU group
    (./handler/DDP/utils.py)
    DDP requires to setup GPU group, which can broadcast weights to all GPUs.
    This function set tcp connection between processes.
    Implement initialize_group function.

    you should use
    1. dist.init_process_group() for tcp connection
    2. torch.cuda.set_device() for setting device
    """
    dist_url = f"tcp://{host}:{port}"
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=num_gpu,
    )
    torch.cuda.set_device(proc_id % num_gpu)


def destroy_process():
    """Problem 6: Destroy GPU group
    (./handler/DDP/utils.py)
    Implement destroy_process function.
    Just call the torch.distributed's destroy function.
    """
    dist.destroy_process_group()
