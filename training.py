import time
import random
import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from network import ClassificationNetwork
from network import RegressionNetwork
from dataset import get_dataloader

def train_regression_dagger(model, training_data, training_labels, dagitr):
    gpu = torch.device('cuda')

    infer_action = model.to(gpu)
    infer_action.train()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=0.65e-5)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    nr_epochs = 25
    batch_size = 20
    # nr_of_classes = 9  # needs to be changed
    start_time = time.time()

    indices = torch.randperm(len(training_data))
    # print(type(training_data), training_data) 
    # training_data = torch.FloatTensor(training_data)
    # training_labels = torch.FloatTensor(training_labels)
    print(training_data.size())

    training_data = torch.permute(training_data, (0, 3, 1, 2)).to(gpu)
    training_labels = torch.Tensor.float(training_labels).to(gpu)
    shuffled_data = training_data[indices]
    shuffled_labels = training_labels[indices]

    batch_in = [shuffled_data[i:i + batch_size] for i in range(0, len(shuffled_data), batch_size)]
    batch_gt = [shuffled_labels[i:i + batch_size] for i in range(0, len(shuffled_labels), batch_size)]

    print(batch_in)

    losses = []
    for epoch in range(nr_epochs):
        total_loss = 0

        for batch_in, batch_gt in zip(batch_in, batch_gt):
            print("batching")
            print(batch_in.size())
            batch_out = infer_action(batch_in)
            print(batch_out)

            loss = mse_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        lrBefore = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, total_loss, lrBefore, lrAfter, time_left))
        
        losses.append(total_loss)
        # if total_loss <= 0.01:
        #     break

    # cpuLoss = 
    cpuLoss = losses

    epochs = list(range(nr_epochs))
    print(losses[-1])
    print(epochs)
    print(cpuLoss)
    plt.plot(epochs, cpuLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Training Loss")
    plt.savefig('training_loss_reg' + str(dagitr) + '.png')

    return (losses[-1], infer_action)


def train_classification(data_folder, save_path, lr, use_observations, should_augment):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    gpu = torch.device('cuda')

    infer_action = ClassificationNetwork(use_observations).to(gpu)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=lr)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)

    nr_epochs = 100
    batch_size = 128
    nr_of_classes = 9  # needs to be changed
    start_time = time.time()

    train_loader = get_dataloader(data_folder, batch_size, False, use_observations)
    # valid_loader = get_dataloader(data_folder, batch_size, True, use_observations)

    losses = []
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in_image, batch_in_obs, batch_gt = batch[0][0].to(gpu), batch[0][1].to(gpu), batch[1].to(gpu)

            batch_out = infer_action(batch_in_image, batch_in_obs)
            batch_gt = infer_action.actions_to_classes(batch_gt)

            loss = cross_entropy_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        lrBefore = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, total_loss, lrBefore, lrAfter, time_left))

        losses.append(total_loss)
        if total_loss <= 0.01:
            break

    # for batch_idx, batch in enumerate(valid_loader):
    #     batch_in_image, batch_in_obs, batch_gt = batch[0][0].to(gpu), batch[0][1].to(gpu), batch[1].to(gpu)
    #     batch_out = infer_action(batch_in_image, batch_in_obs)
    #     batch_gt = infer_action.actions_to_classes(batch_gt)


    cpuLoss = [loss.cpu().detach().float() for loss in losses]

    
    torch.save(infer_action, save_path + "classmodel.pth")
    epochs = list(range(nr_epochs))
    print(epochs)
    print(cpuLoss)
    plt.plot(epochs, cpuLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Training Loss")
    plt.savefig('training_loss_class.png')
    plt.show()


def train_regression(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    gpu = torch.device('cuda')

    infer_action = RegressionNetwork().to(gpu)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=0.65e-5)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    nr_epochs = 100
    batch_size = 128
    nr_of_classes = 9  # needs to be changed
    start_time = time.time()

    train_loader = get_dataloader(data_folder, batch_size)

    losses = []
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in_image, batch_in_obs, batch_gt = batch[0][0].to(gpu), batch[0][1].to(gpu), batch[1].to(gpu)

            batch_out = infer_action(batch_in_image, batch_in_obs)

            loss = mse_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        lrBefore = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, total_loss, lrBefore, lrAfter, time_left))
        
        losses.append(total_loss)
        if total_loss <= 0.01:
            break

    cpuLoss = [loss.cpu().detach().float() for loss in losses]

    torch.save(infer_action.state_dict(), save_path + "regmodel.pth")
    epochs = list(range(nr_epochs))
    print(epochs)
    print(cpuLoss)
    plt.plot(epochs, cpuLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Training Loss")
    plt.savefig('training_loss_reg.png')
    plt.show()    

def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """

    #loss = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
    batch_gt = torch.tensor(batch_gt).cuda()
    lpred = torch.log(batch_out)
    ytruelogpred = torch.mul(batch_gt, lpred)
    loss_tensor = -torch.mean(torch.sum(ytruelogpred, dim=1))

    return loss_tensor

def mse_loss(batch_out, batch_gt):
    loss = torch.nn.MSELoss()
    return loss(batch_out, batch_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC518 Homework1 Imitation Learning')
    parser.add_argument('-t', '--network_type', default="c", type=str, help='type of network, c for class, r for regress, mc')
    parser.add_argument('-l', '--learning_rate', default="0.00001", type=float, help='learning rate')
    parser.add_argument('-o', '--use_observations', default=False, type=bool, help='use other sensors')
    parser.add_argument('-a', '--augment', default=False, type=bool, help='augment data')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    if args.network_type == "c":
        train_classification(args.data_folder, args.save_path, args.learning_rate, args.use_observations, args.augment)
    elif args.network_type == "r":
        train_regression(args.data_folder, args.save_path, args.learning_rate, args.use_observations, args.augment)
