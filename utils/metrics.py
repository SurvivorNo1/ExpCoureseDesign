import torch
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

# 计算准确率
def cal_acc(y_pred,y_true):
    # 比较预测标签和真实标签并计算准确度
    correct = torch.sum(y_pred == y_true).item()
    total = y_true.shape[0]
    accuracy = correct / total
    return accuracy
# 计算精确率
def cal_precision(y_pred,y_true):
    # 计算精确率
    precision = precision_score(y_true, y_pred,average='macro',zero_division=0)
    return precision
# 计算召回率
def cal_recall(y_pred,y_true):
    # 计算召回率
    recall = recall_score(y_true, y_pred,average='macro',zero_division=0)
    return recall