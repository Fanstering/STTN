import os
import json
import argparse
import datetime
import numpy as np
from shutil import copyfile
import torch
import torch.multiprocessing as mp

from core.trainer import Trainer
from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

## 设置命令行运行的参数
parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs/davis.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
args = parser.parse_args()


## 主运行程序
def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank  ## 非分布式 则都赋值为-1
        ## 采用分布式训练
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank'])))

    config['save_dir'] = os.path.join(config['save_dir'], '{}_{}'.format(config['model'],
                                                                         os.path.basename(args.config).split('.')[0]))
    ## 是否采用GPU
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'
    # 非分布式训练
    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)  ## 递归创建目录 exist_ok=True表示已存在时不报错
        config_path = os.path.join(  ## os.path.join(path1[, path2[, ...]])	把目录和文件名合成一个路径
            config['save_dir'], config['config'].split('/')[-1])
        if not os.path.isfile(config_path):  ## 如果没有
            copyfile(config['config'], config_path)  ## shutil.copyfile(src,tar)
        print('[**] create folder {}'.format(config['save_dir']))  ## release中没有就创建一个

    trainer = Trainer(config, debug=args.exam)  ## 参数-e 只要出现就赋值为true  也就是命令行有-e就执行debug
    trainer.train()


if __name__ == "__main__":

    # loading configs
    config = json.load(open(args.config))
    config['model'] = args.model
    config['config'] = args.config

    ## 是否分布式GPU
    # setting distributed configurations
    config['world_size'] = get_world_size()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1":
        # manually launch distributed processes 
        # mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
        main_worker(0,config)
    else:
        # multiple processes have been launched by openmpi 
        config['local_rank'] = get_local_rank()
        config['global_rank'] = get_global_rank()
        main_worker(-1, config)
