import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from model import SVM
from model import Decisiontree
from model.LeNet import LeNet
from dataset.Mnist import MnistDataset
from utils.metrics import cal_acc

TRAIN_DATAPATH = "./data/MNIST/experiment_09_training_set.csv"
TEST_DATAPATH = "./data/MNIST/experiment_09_testing_set.csv"
LR_RATE = 0.1 # 学习率
OPTIMIZER = "SGD" # 可选"SGD","Adam","RMSprop"
BATCH_SIZE = 64 # 批大小
OUTPUT_PATH = "./weights/" # 模型输出路径
IMAGE_PATH = "./images/" # 训练过程可视化输出路径
EPOCHS = 10 # 训练轮数
LOG_ITER = 50 # 训练过程中的日志输出间隔

def draw_loss(loss):
    title = "{}_lr_{}".format(OPTIMIZER,LR_RATE)
    plt.figure()
    plt.plot(loss)
    plt.title(title)
    plt.xlabel("iter/{}".format(LOG_ITER) + " (per {} iter)".format(LOG_ITER))
    plt.ylabel("loss")
    save_path = IMAGE_PATH + title + ".png"
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
    plt.savefig(save_path)
    plt.close()
    
def set_model():
    model = LeNet()
    return model

def set_optim(model):
    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LR_RATE)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    elif OPTIMIZER == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=LR_RATE)
    else:
        raise NotImplementedError("Optimizer is not implemented.")
    return optimizer

def set_criterion():
    criterion = nn.MSELoss()
    return criterion

def set_dataloader():
    train_data = pd.read_csv(TRAIN_DATAPATH)
    test_data = pd.read_csv(TEST_DATAPATH)
    train_data = np.array(train_data).astype(np.float32)
    test_data = np.array(test_data).astype(np.float32)

    train_x = train_data[:,1:].reshape(-1,28,28)
    train_y = train_data[:,0].reshape(-1,1).astype(np.int32)

    train_y = np.eye(10)[train_y.reshape(-1)].astype(np.float32)

    test_x = test_data[:,1:].reshape(-1,28,28)
    test_y = test_data[:,0].reshape(-1,1).astype(np.int32)
    test_y = np.eye(10)[test_y.reshape(-1)].astype(np.float32)
    # 定义数据预处理
    traintransforms = transforms.Compose([
                                            transforms.ToTensor(),
                                        ])
    # 创建NumpyDataset实例
    traindataset = MnistDataset(train_x, train_y, transform=traintransforms)
    testdataset = MnistDataset(test_x, test_y, transform=traintransforms)
    # 创建DataLoader
    trainloader = data.DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = data.DataLoader(testdataset, batch_size=len(testdataset), shuffle=True)
    return trainloader, testloader

def train(model,opt,criterion,trainloader,testloader):
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        for i,(x,y) in enumerate(trainloader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat,y)
            if i%LOG_ITER == 0:
                print("iter: {}, loss: {}".format(i,loss))
                losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_acc = 0
            for x,y in trainloader:
                y_hat = model(x)
                y_hat = y_hat.argmax(dim=1).type_as(y)
                y = y.argmax(dim=1)
                train_acc += cal_acc(y_hat,y)
            train_acc /= len(trainloader)
            test_acc = 0
            for x,y in testloader:
                y_hat = model(x)
                y_hat = y_hat.argmax(dim=1).type_as(y)
                y = y.argmax(dim=1)
                test_acc += cal_acc(y_hat,y)
            test_acc /= len(testloader)
            print("epoch: {}, train_acc: {}, test_acc: {}".format(epoch,train_acc,test_acc))
    draw_loss(losses)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    torch.save(model.state_dict(),OUTPUT_PATH + "{}_lr_{}_model.pth".format(OPTIMIZER,LR_RATE))
    
if __name__ =="__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model', type=str, default='LeNet', choices=['LeNet', 'SVM', 'Decisiontree'], help='Model name')
    parser.add_argument('--train_datapath', type=str, default='./data/MNIST/experiment_09_training_set.csv', help='Path to training dataset')
    parser.add_argument('--test_datapath', type=str, default='./data/MNIST/experiment_09_testing_set.csv', help='Path to testing dataset')
    parser.add_argument('--lr_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'], help='Optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_path', type=str, default='./weights/', help='Model output path')
    parser.add_argument('--image_path', type=str, default='./images/', help='Visualization output path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--log_iter', type=int, default=100, help='Log iteration')
    # 解析命令行参数
    args = parser.parse_args()
    TRAIN_DATAPATH = args.train_datapath
    TEST_DATAPATH = args.test_datapath
    LR_RATE = args.lr_rate
    OPTIMIZER = args.optimizer
    BATCH_SIZE = args.batch_size
    OUTPUT_PATH = args.output_path # 模型输出路径
    IMAGE_PATH = args.image_path # 可视化输出路径
    EPOCHS = args.epochs # 训练轮数
    LOG_ITER = args.log_iter # 日志打印间隔
    # 开始训练
    trainloader, testloader = set_dataloader()
    if args.model == 'LeNet':
        model = set_model()
        optimizer = set_optim(model)
        criterion = set_criterion()
        train(model,optimizer,criterion,trainloader,testloader)
    elif args.model == 'SVM':
        SVM.train(OUTPUT_PATH,trainloader)
    elif args.model == 'Decisiontree':
        Decisiontree.train(OUTPUT_PATH,trainloader)
    else:
        raise NotImplementedError("Model is not supported.")