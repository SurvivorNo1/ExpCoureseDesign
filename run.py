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
    log_file = f"./logs/LeNet_{i+1}.log"
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