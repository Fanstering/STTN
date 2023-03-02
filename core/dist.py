import os
import io
import re
import subprocess
import logging
import random
import torch
import numpy as np


## 分布式训练所用到的函数集合 distribute
def get_world_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    if os.environ.get('PMI_SIZE') is not None:  ## os.environ.get() 获取系统环境变量 参数就是环境变量名
        return int(os.environ.get('PMI_SIZE') or 1)
    elif os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
    else:
        return torch.cuda.device_count()


def get_global_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    if os.environ.get('PMI_RANK') is not None:
        return int(os.environ.get('PMI_RANK') or 0)
    elif os.environ.get('OMPI_COMM_WORLD_RANK') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
    else:
        return 0


def get_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    if os.environ.get('MPI_LOCALRANKID') is not None:
        return int(os.environ.get('MPI_LOCALRANKID') or 0)
    elif os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)
    else:
        return 0


def get_master_ip():
    if os.environ.get('AZ_BATCH_MASTER_NODE') is not None:
        return os.environ.get('AZ_BATCH_MASTER_NODE').split(':')[0]
    elif os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE') is not None:
        return os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE')
    else:
        return "127.0.0.1"