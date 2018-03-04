import os
import time
import random
import numpy as np
import argparse
import logging
import functools
import shutil

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local packages
from tool.utils import *

# debug
import ipdb

def main():
    # get configuration
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        cfg = parse_yaml(p.config)

    # set randomness
    if cfg.seed==-1:
        seed = random.randint(1, 10000)
    else:
        seed = cfg.seed
    print(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cfg.cuda:
        torch.cuda.manual_seed_all(seed)

    # dataset and dataloader
    train_dataset = dynamic_import(cfg.dataset_train).Dataset()
    train_dataset.initialize(cfg.dataset_train_args)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=cfg.workers)

    test_dataset = train_dataset #TODO: real test set
    test_dataloader = train_dataloader

    # network
    model_cfg = dynamic_import(cfg.model).create_model_config(cfg, train_dataset)
    net = dynamic_import(cfg.model).Model(model_cfg)
    net.double()
    if cfg.cuda:
        net = net.cuda()
    print(net)
    
    # loss
    criterion = getattr(nn, cfg.loss)()
    if cfg.cuda: #TODO
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)

    # load pretrained model
    start_epoch = 0
    if cfg.model_path is not None:
        ckpt = torch.load(cfg.model_path)
        net.load_state_dict(ckpt["net_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"]

    # define training process
    trainer = dynamic_import(cfg.trainer).train
    train_cfg = dynamic_import(cfg.trainer).create_train_config(cfg, train_dataset)
    train_an_epoch = functools.partial(trainer, 
                                       dataloader=train_dataloader,
                                       net=net,
                                       criterion=criterion,
                                       optimizer=optimizer,
                                       cfg=train_cfg)

    # define testing process
    tester = dynamic_import(cfg.tester).test
    test_cfg = dynamic_import(cfg.tester).create_test_config(cfg, test_dataset)
    test = functools.partial(tester, 
                             dataloader=test_dataloader,
                             net=net,
                             criterion=criterion,
                             cfg=test_cfg)

    # specify output workspace
    out_config_path = validate_path(cfg.save_dir, "config.yaml")
    shutil.copy(p.config, out_config_path)
    out_log_dir = validate_dir(cfg.save_dir, "log")
    out_ckpt_dir = validate_dir(cfg.save_dir, "ckpt")
    logging.basicConfig(filename=out_log_dir+"/log.txt",
                        filemode='a',
                        level=logging.INFO)

    # start
    for epoch in range(start_epoch, cfg.n_epoch+1):
        train_an_epoch(epoch=epoch)

        # save checkpoint
        if epoch % cfg.save_step == 0:
            ckpt = {"epoch": epoch,
                    "net_state": net.state_dict(),
                    "opt_state": optimizer.state_dict()}
            ckpt_path = os.path.join(out_ckpt_dir,"ep-{}.pt".format(epoch))
            torch.save(ckpt, ckpt_path)
            print("Save checkpoint to {}".format(ckpt_path))
            
        # online evaluation
        if epoch % cfg.eval_step == 0:
            test()
        
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file (.yaml)")

    return parser

if __name__=="__main__":
    main()
