import argparse
import numpy as np
import torch
import os
import seaborn as sn
import matplotlib.pyplot as plt

from train import set_dataloader
from model.LeNet import LeNet
from utils.metrics import cal_precision, cal_recall
from model import SVM
from model import Decisiontree
from sklearn.metrics  import confusion_matrix, ConfusionMatrixDisplay
import joblib

OUTPUT_PATH = "./weights/" # 模型输出路径
IMAGE_PATH = "./images/" # 训练过程可视化输出路径

def draw_confusion_matrix(y, y_hat, name):
    precision = cal_precision(y, y_hat)
    recall = cal_recall(y, y_hat)
    F1 = 2 * precision * recall / (precision + recall)
    print("model config: {} \nprecision: {:.4f}, recall: {:.4f}, F1: {:.4f}".format(name,precision, recall, F1))
    
    fig = plt.figure(figsize=(11, 11))
    cm = confusion_matrix(y, y_hat)
    cm = ConfusionMatrixDisplay(cm, display_labels=range(10))
    cm.plot(values_format='d', cmap='Blues')
    savepath = IMAGE_PATH+"cm_"+name+".png"
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
    plt.savefig(savepath)
    plt.close(fig)
    


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--output_path', type=str,
                        default='./weights/', help='Model output path')
    parser.add_argument('--image_path', type=str,
                        default='./images/', help='Visualization output path')
    parser.add_argument('--model', type=str, default='LeNet', choices=['LeNet', 'SVM', 'Decisiontree'], help='Model name')
    
    args = parser.parse_args()
    OUTPUT_PATH = args.output_path # 模型输出路径
    IMAGE_PATH = args.image_path # 可视化输出路径
    # 设置测试数据
    trainloader, testloader = set_dataloader()
    # 模型权重路径
    weightspaths = os.listdir(OUTPUT_PATH)
    # 测试模型
    if args.model == 'LeNet':
        for weightpath in weightspaths:
            weightpath = os.path.join(OUTPUT_PATH,weightpath)
            if weightpath.split('.')[-1] == 'pth':
                model = LeNet()
                model.load_state_dict(torch.load(weightpath))
                y_hat, y = LeNet.test(model,testloader)
                draw_confusion_matrix(y, y_hat,weightpath.split('/')[-1])    
    elif args.model == 'SVM':
        # 测试svm
        model = joblib.load(args.output_path+"svm.pkl")
        y_hat, y = SVM.test(model,testloader)
        draw_confusion_matrix(y, y_hat,"svm")
    elif args.model == 'Decisiontree':
        # 测试决策树
        model = joblib.load(args.output_path+"decisiontree.pkl")
        y_hat, y = Decisiontree.test(model,testloader)
        draw_confusion_matrix(y, y_hat,"decisiontree")
    else:
        raise Exception("No model named {}".format(args.model))

