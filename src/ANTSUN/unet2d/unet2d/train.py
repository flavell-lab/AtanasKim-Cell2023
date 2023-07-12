import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import copy
import time
import csv
import os

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


def train_epoch(model, device, data_loader, f_loss, f_metric, optimizer, loss_weighting, metric_weighting):
    u_batch_time = AverageMeter("Batch", ":6.3f")
    u_data_time = AverageMeter("Data", ":6.3f")
    u_loss = AverageMeter("Loss", ":.4e")
    u_metric = AverageMeter("Metric", ":.4e")

    model.train()

    t_start = time.time()
    t_end = time.time()
    for sample in data_loader["train"]:
        t_start = time.time()
        u_data_time.update(t_start - t_end)

        # data to GPU if needed
        x = sample["x"]
        y_true = sample["y_true"]
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)
        if loss_weighting or metric_weighting:
            weights = sample["weights"]
            weights = weights.to(device, non_blocking=True)
                        
        # forward pass
        y = model(x)
        if loss_weighting:
            loss = f_loss(y, y_true, weights)
        else:
            loss = f_loss(y, y_true)
        
        if metric_weighting:
            metric = f_metric(y, y_true, weights)
        else:
            metric = f_metric(y, y_true)

        u_metric.update(metric.item(), x.size(0))
        u_loss.update(loss.item(), x.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # elapsed time
        u_batch_time.update(time.time() - t_start)
        t_end = time.time()

    return u_batch_time, u_data_time, u_loss, u_metric


def validate_epoch(model, device, data_loader, f_loss, f_metric, loss_weighting, metric_weighting):
    u_batch_time = AverageMeter("Batch", ":6.3f")
    u_data_time = AverageMeter("Data", ":6.3f")
    u_loss = AverageMeter("Loss", ":.4e")
    u_metric = AverageMeter("Metric", ":.4e")

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        t_start = time.time()
        t_end = time.time()
        for sample in data_loader["val"]:
            t_start = time.time()
            u_data_time.update(t_start - t_end)

            # data to GPU if needed
            x = sample["x"]
            y_true = sample["y_true"]
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            if loss_weighting or metric_weighting:
                weights = sample["weights"]
                weights = weights.to(device, non_blocking=True)

            # forward pass
            y = model(x)
            if loss_weighting:
                loss = f_loss(y, y_true, weights)
            else:
                loss = f_loss(y, y_true)
            if metric_weighting:
                metric = f_metric(y, y_true, weights)
            else:
                metric = f_metric(y, y_true)
                
            u_metric.update(metric.item(), x.size(0))
            u_loss.update(loss.item(), x.size(0))

            # elapsed time
            u_batch_time.update(time.time() - t_start)
            t_end = time.time()

    return u_batch_time, u_data_time, u_loss, u_metric


def log_to_csv(path_log_csv, csv_log_stack):
    with open(path_log_csv, "a") as csvf:
        writer = csv.writer(csvf, delimiter=",")
        while len(csv_log_stack) > 0:
            writer.writerow(csv_log_stack.pop(0))


def train_model(model, device, data_loader, f_loss, f_metric, optimizer, scheduler, args,
                loss_weighting=False, metric_weighting=False):
    start_epoch = args["start_epoch"]
    n_epochs = args["n_epochs"]
    print_freq = args["print_freq"]
    path_checkpoint = args["path_checkpoint"]
    checkpoint_freq = args["checkpoint_freq"]
    path_log_tb = args["path_log_tb"]
    path_log_csv = args["path_log_csv"]
    
    # model best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # csv log
    if not os.path.isfile(path_log_csv):
        with open(path_log_csv, "a") as csvf:
            writer = csv.writer(csvf, delimiter=",")
            writer.writerow(["epoch", "train_loss", "val_loss",
                             "train_metric", "val_metric", "lr"])
    csv_log_stack = []
    
    # train loop
    best_loss = float("inf")
    t_start = time.time()
    
    with SummaryWriter(log_dir=path_log_tb) as tb_writer:
        for epoch in range(start_epoch, n_epochs):
            # train
            train_batch_t, train_data_t, train_loss, train_metric = train_epoch(
                model, device, data_loader, f_loss, f_metric, optimizer,
                loss_weighting, metric_weighting)

            # validate
            val_batch_t, val_data_t, val_loss, val_metric = validate_epoch(
                model, device, data_loader, f_loss, f_metric,
                loss_weighting, metric_weighting)

            # lr
            if scheduler:
                scheduler.step(val_loss.avg)                

            # save history
            train_loss_ = train_loss.avg
            val_loss_ = val_loss.avg
            train_metric_ = train_metric.avg
            val_metric_ = val_metric.avg
            lr_ = optimizer.param_groups[0]["lr"]

            csv_log_stack.append([epoch, train_loss_, val_loss_, train_metric_,
                                  val_metric_, lr_])
            
            tb_writer.add_scalar('Loss/train', train_loss_, epoch)
            tb_writer.add_scalar('Loss/val', val_loss_, epoch)
            tb_writer.add_scalar('Metric/train', train_metric_, epoch)
            tb_writer.add_scalar('Metric/val', val_metric_, epoch)
            tb_writer.add_scalar('learning rate', lr_, epoch)

            # print
            if (epoch % print_freq) == 0:
                print("Epoch {}/{}".format(epoch, n_epochs - 1), end=" - ")
                print('train t: {:.3f}'.format(train_batch_t.sum), end=" ")
                print('val t: {:.3f}'.format(val_batch_t.sum), end=" ")
                print('train loss: {:.4f}'.format(train_loss.avg), end=" ")
                print('val loss: {:.4f}'.format(val_loss.avg))

            # checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                # save model/optimizer state
                torch.save({
                    "epoch": epoch,
                    'model_state_dict': model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, path_checkpoint)
                # log to csv
                log_to_csv(path_log_csv, csv_log_stack)

            # save weights if best
            epoch_loss = val_loss.avg
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    t_end = time.time()
    print("Total elapsed: {:.3f}".format(t_end - t_start))
    print("Mean time per epoch: {:.3f}".format((t_end - t_start) / (n_epochs - start_epoch)))
    print("Best loss: {:.3f}".format(best_loss))

    # log to csv
    log_to_csv(path_log_csv, csv_log_stack)
    
    return best_model_wts
