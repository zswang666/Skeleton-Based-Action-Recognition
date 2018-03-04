import torch
from torch.autograd import Variable
from collections import namedtuple
import time
import logging
#from logger import Logger as tf_logger

logger = logging.getLogger(__name__)

def train(epoch, dataloader, net, criterion, optimizer, cfg):
    # set training flag
    net.train()
    for i, data in enumerate(dataloader, 0):
        tic = time.time()

        # unpack data
        annotation = data[0]
        adj_matrix = data[1]
        target = data[2]
        seq_length = data[3]

        # prepare data
        padding = torch.zeros(len(annotation), cfg.max_frames, cfg.n_joints, cfg.state_dim - cfg.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)
        if cfg.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()
            seq_length = seq_length.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)
        seq_length = Variable(seq_length)

        # inference
        output = net(init_input, annotation, adj_matrix, seq_length)

        # compute loss
        loss = criterion(output, target)

        # backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()

        # log
        if i % int(len(dataloader) / 10 + 1) == 0 and cfg.verbose:
            log = "[%d/%d][%d/%d] Loss: %.4f" % (epoch, cfg.n_epoch, i, len(dataloader), loss.data[0])
            logger.info(log)
            print(log+" Elapsed time: {:.4f}".format(toc-tic))
            
            # tensorboard
            '''
            info = {
                "loss": loss.data[0]    
            }
            global_step = epoch*len(dataloader) + i
            for tag, v in info.items():
                tf_logger.scalar_summary(tag, v, global_step)
            '''

def create_train_config(cfg, dataset):
    train_cfg = dict()
    train_cfg["n_joints"] = dataset.n_joints
    train_cfg["state_dim"] = cfg.model_args.state_dim
    train_cfg["annotation_dim"] = cfg.dataset_train_args.annotation_dim
    train_cfg["cuda"] = cfg.cuda
    train_cfg["verbose"] = cfg.verbose
    train_cfg["n_epoch"] = cfg.n_epoch
    train_cfg["max_frames"] = dataset.max_frames
    train_cfg = namedtuple("GenericDict", train_cfg.keys())(**train_cfg)

    return train_cfg
