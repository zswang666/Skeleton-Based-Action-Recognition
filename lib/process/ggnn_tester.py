import torch
from torch.autograd import Variable
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

def test(dataloader, net, criterion, cfg):
    test_loss = 0
    correct = 0
    #net.eval()
    for i, data in enumerate(dataloader, 0):
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

        # compute loss and accuracy
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataloader.dataset)
    log = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          test_loss, correct, len(dataloader.dataset),
          100. * correct / len(dataloader.dataset))
    logger.info(log)
    print(log)

def create_test_config(cfg, dataset):
    train_cfg = dict()
    train_cfg["n_joints"] = dataset.n_joints
    train_cfg["state_dim"] = cfg.model_args.state_dim
    train_cfg["annotation_dim"] = cfg.dataset_train_args.annotation_dim
    train_cfg["cuda"] = cfg.cuda
    train_cfg["max_frames"] = dataset.max_frames
    train_cfg = namedtuple("GenericDict", train_cfg.keys())(**train_cfg)

    return train_cfg
