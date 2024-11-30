import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from pprint import pprint
from embedding_net import EmbeddingNet
from deepgesturedataset import DeepGestureDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, model, optimizer, device):
    # zero gradients
    optimizer.zero_grad()

    # reconstruction loss
    model.to(device)
    feat, recon_data = model(target_data)
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optimizer.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict


def main(args, gesture_dim, n_frames, device):
    dataset = DeepGestureDataset(dataset_file=args.dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)
    print("Total train loader ", len(train_loader))
    print("Start training with print_interval", print_interval)

    model = EmbeddingNet(gesture_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))

    epoch = 0
    i = 0
    # training
    for epoch in range(args.epoch):
        if epoch % 50 == 0:
            print('Epoch {}/{}'.format(epoch + 1, args.epoch))

        for i, batch in enumerate(train_loader, 0):
            batch_gesture = np.asarray(batch, np.float32)
            target_vec = torch.Tensor(batch_gesture).to(device)
            loss = train_iter(target_vec, model, gen_optimizer, device)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], args.batch_size)

            # print training status
            if epoch % 1000 == 0 and (i + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, i + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

        if epoch % 100000 == 0:
            state_dict = model.cpu().state_dict()
            file_path = f'output/embedding_network_{gesture_dim}_{n_frames}_{epoch}.pth'
            print("Saving checkpoint to {}".format(file_path))
            torch.save({'gesture_dim': gesture_dim, 'n_frames': n_frames, 'embedding_model': state_dict}, file_path)

    # save model
    state_dict = model.cpu().state_dict()
    file_path = f'output/embedding_network_{gesture_dim}_{n_frames}_final.pth'
    print("Saving checkpoint to {}".format(file_path))
    torch.save({'gesture_dim': gesture_dim, 'n_frames': n_frames, 'embedding_model': state_dict}, file_path)

    print_summary = 'EP {} ({:3d}) | '.format(epoch, i + 1)
    for loss_meter in loss_meters:
        if loss_meter.count > 0:
            print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
            loss_meter.reset()
    print(print_summary)


if __name__ == '__main__':
    """
    python train_embedding.py --dataset=../data/real_dataset.npz --gpu=cuda:0
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dataset', '-d', required=True, default="../data/real_dataset.npz",
                        help="")
    parser.add_argument('--gpu', '-gpu', required=True, default="cuda:0",
                        help="")
    parser.add_argument('--epoch', '-epoch', type=int, default=500000,
                        help="")
    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help="")

    args = parser.parse_args()
    pprint(args)

    n_frames = 88
    # gesture_dim = 1141
    gesture_dim = 225
    device = torch.device(args.gpu)

    main(args, gesture_dim, n_frames, device)
