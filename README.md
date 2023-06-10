# 环境准备
## 1、进入工作目录ExpCoureseDesign
cd ./ExpCoureseDesign/
## 2、安装依赖
```
pip install -r requirements.txt
```
# 快速开始
## 1、默认已经完成了各个模型的训练,可以直接运行test.py完成模型的测试
分别运行以下脚本
```shell
python  test.py --model LeNet
python  test.py --model SVM
python  test.py --model Decisiontree
```
运行结果：
```shell
model config: Adam_lr_0.0001_model.pth 
precision: 0.9761, recall: 0.9771, F1: 0.9766
model config: Adam_lr_0.001_model.pth 
precision: 0.9725, recall: 0.9731, F1: 0.9728
model config: Adam_lr_0.01_model.pth 
precision: 0.1000, recall: 0.0099, F1: 0.0180
model config: RMSprop_lr_0.0001_model.pth 
precision: 0.9776, recall: 0.9779, F1: 0.9778
model config: RMSprop_lr_0.001_model.pth 
precision: 0.9784, recall: 0.9811, F1: 0.9797
model config: RMSprop_lr_0.01_model.pth 
precision: 0.1000, recall: 0.0102, F1: 0.0186
model config: SGD_lr_0.0001_model.pth 
precision: 0.4202, recall: 0.2935, F1: 0.3456
model config: SGD_lr_0.001_model.pth 
precision: 0.7121, recall: 0.5785, F1: 0.6384
model config: SGD_lr_0.01_model.pth 
precision: 0.9644, recall: 0.9628, F1: 0.9636
model config: svm 
precision: 0.9704, recall: 0.9710, F1: 0.9707
model config: decisiontree 
precision: 0.8676, recall: 0.8670, F1: 0.8673
```
## 2、若想要重新训练，可以直接运行run.py,快速完成训练、测试，得到所有结果
注意：使用了多进程训练，运行时可能会卡顿
```shell
python ./run.py
```

```python
import subprocess

scripts = ["python  train.py --model LeNet --lr_rate 0.01 --optimizer SGD",
            "python train.py --model LeNet --lr_rate 0.001 --optimizer SGD",
            "python train.py --model LeNet --lr_rate 0.0001 --optimizer SGD",
            "python train.py --model LeNet --lr_rate 0.01 --optimizer Adam",
            "python train.py --model LeNet --lr_rate 0.001 --optimizer Adam",
            "python train.py --model LeNet --lr_rate 0.0001 --optimizer Adam",
            "python train.py --model LeNet --lr_rate 0.01 --optimizer RMSprop",
            "python train.py --model LeNet --lr_rate 0.001 --optimizer RMSprop",
            "python train.py --model LeNet --lr_rate 0.0001 --optimizer RMSprop"]

# 存储子进程的列表
processes = []

# 启动子进程
for i, script in enumerate(scripts):
    log_file = f"./logs/model_{i+1}.log"
    process = subprocess.Popen(script,stdout=open(log_file, "w+"), shell=True,bufsize=0)
    processes.append(process)

# 等待所有子进程结束
for process in processes:
    process.wait()
    
subprocess.run('python  train.py --model Decisiontree', shell=True)
subprocess.run('python  train.py --model SVM', shell=True) 

subprocess.run('python test.py --model LeNet', shell=True)
subprocess.run('python test.py --model Decisiontree', shell=True)
subprocess.run('python test.py --model SVM', shell=True)
```

***
# 下面是具体过程分步解析
# 训练模型
## 1、可选参数
```python
parser.add_argument('--train_datapath', type=str, default='../data/MNIST/experiment_09_training_set.csv', help='Path to training dataset')
parser.add_argument('--test_datapath', type=str, default='../data/MNIST/experiment_09_testing_set.csv', help='Path to testing dataset')
parser.add_argument('--lr_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'], help='Optimizer')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--output_path', type=str, default='./weights/', help='Model output path')
parser.add_argument('--image_path', type=str, default='./images/', help='Visualization output path')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--log_iter', type=int, default=100, help='Log iteration')
```
### 示例值
```shell
# 所有可选参数（示例值）
TRAIN_DATAPATH = "../data/MNIST/experiment_09_training_set.csv"
TEST_DATAPATH = "../data/MNIST/experiment_09_testing_set.csv"
LR_RATE = 0.1 # 学习率
OPTIMIZER = "SGD" # 可选"SGD","Adam","RMSprop"
BATCH_SIZE = 64 # 批大小
OUTPUT_PATH = "./weights/" # 模型输出路径
IMAGE_PATH = "./images/" # 训练过程可视化输出路径
EPOCHS = 10 # 训练轮数
LOG_ITER = 100 # 训练过程中的日志输出间隔
```
## 2、分别运行以下脚本
```
python train.py --lr_rate 0.01 --optimizer SGD
python train.py --lr_rate 0.001 --optimizer SGD
python train.py --lr_rate 0.0001 --optimizer SGD
python train.py --lr_rate 0.01 --optimizer Adam
python train.py --lr_rate 0.001 --optimizer Adam
python train.py --lr_rate 0.0001 --optimizer Adam
python train.py --lr_rate 0.01 --optimizer RMSprop
python train.py --lr_rate 0.001 --optimizer RMSprop
python train.py --lr_rate 0.0001 --optimizer RMSprop
python  train.py --model Decisiontree
python  train.py --model SVM
```
得到6个模型的权重以及对应的训练loss变化图,并得到svm和决策树的两个模型
# 模型验证
## 1、可选参数
```python
parser.add_argument('--output_path', type=str,
                    default='./weights/', help='Model output path')
parser.add_argument('--image_path', type=str,
                    default='./images/', help='Visualization output path')
parser.add_argument('--model', type=str, default='LeNet', choices=['LeNet', 'SVM', 'Decisiontree'], help='Model name')
```
## 2、运行以下脚本
```
python  test.py --model LeNet --output_path  ./weights/ --image_path ./images/
python  test.py --model SVM   --output_path  ./weights/ --image_path ./images/
python  test.py --model Decisiontree --output_path  ./weights/ --image_path ./images/
```
加载前面训练的模型权重并绘制对应的混淆矩阵、计算precision、recall、F1。

