# -*- encoding:utf-8 -*-
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, classification_report
logging.getLogger().setLevel(logging.INFO)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# ****************************************************/
#     导入数据集
# ****************************************************/
def resize(mylist,length):
    if len(mylist) >= length:
        mylist = mylist[0:length]
    else:
        mylist.extend([0.0]*(length-len(mylist)))
    return mylist

def read_series(file,length):
    with open(file,mode='r',encoding='utf-8') as f:
        lines = list(f.readlines())
        temp_list = lines[0].strip().replace('[','').replace(']','').split(',')
        timestamps = [float(i) for i in temp_list if i!='']
        first = timestamps[0]
        timestamps = [i-first for i in timestamps]
        temp_list = lines[3].strip().replace('[','').replace(']','').split(',')
        lengths = [abs(float(i)) for i in temp_list if i!='']
        temp_list = lines[4].strip().replace('[','').replace(']','').split(',')
        s2c_lengths = [abs(float(i)) for i in temp_list if i!='']
        temp_list = lines[5].strip().replace('[','').replace(']','').split(',')
        c2s_lengths = [abs(float(i)) for i in temp_list if i!='']
    timestamps = resize(timestamps,length)
    lengths = resize(lengths,length)
    s2c_lengths = resize(s2c_lengths,length)
    c2s_lengths = resize(c2s_lengths,length)
    label_dict={'amplifier':0,'auto-response':1,'chat':5,'hybrid':3,'publish':4,'report':2}
    label = 0
    for k in label_dict.keys():
        if k in file:
            label = label_dict[k]
    if (timestamps is None) or (lengths is None):
        return None,None
    if s2c_lengths is None:
        s2c_lengths = [0.0] * length
    if c2s_lengths is None:
        c2s_lengths = [0.0] * length
    res = np.array([timestamps,lengths,s2c_lengths,c2s_lengths],dtype='float32').T
    # print(len(timestamps),len(lengths),len(s2c_lengths),len(c2s_lengths))
    return res,label

# raw_df = np.DataFrame(columns=['label','time','len','s2c','c2s'])

text_path = '/home/zhaihaonan/twitterbot_seq/series'
length = 128
def read_dataset(text_path,length):
    sequences = list()
    labels = list()
    # label_dict={'amplifier':0,'auto-response':1,'chat':2,'hybrid':3,'publish':4,'report':5}
    for root, dirs, files in os.walk(text_path):
        for file in files:
            file_path = os.path.join(root, file)
            s,l = read_series(file_path,length)
            if s is None:
                continue
            if l == 5:
                continue
            sequences.append(s)
            labels.append(l)
    return sequences,labels

sequences,labels = read_dataset(text_path,length)
# ----------------------------------------------------#
#   数据集划分
# ----------------------------------------------------#
# 将numpy数组转换为PyTorch张量
# final_seq = torch.tensor(final_seq, dtype=torch.float)

def split_and_convert(arr_list, label_list):
    num_samples = len(arr_list)
    indices = list(range(num_samples))
    random.shuffle(indices)

    num_train = int(0.8 * num_samples)

    # 按照随机打乱后的索引进行划分
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_arr_list = [arr_list[i] for i in train_indices]
    train_label_list = [label_list[i] for i in train_indices]
    test_arr_list = [arr_list[i] for i in test_indices]
    test_label_list = [label_list[i] for i in test_indices]

    # 转换为 PyTorch 张量
    train_arr_tensor = torch.from_numpy(np.array(train_arr_list))
    train_label_tensor = torch.from_numpy(np.array(train_label_list))
    test_arr_tensor = torch.from_numpy(np.array(test_arr_list))
    test_label_tensor = torch.from_numpy(np.array(test_label_list))

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_arr_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_arr_tensor, test_label_tensor)

    return train_dataset, test_dataset

train_dataset,test_dataset = split_and_convert(sequences,labels)
logging.info(len(train_dataset))
logging.info(len(test_dataset))
'''
/****************************************************/
    网络模型
/****************************************************/
'''
# ----------------------------------------------------#
#   LSTM 模型
# ----------------------------------------------------#
class TimeSeriesClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=256, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)  # output_size classes

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 只取LSTM输出中的最后一个时间步
        x = self.fc(x)  # 通过一个全连接层
        return x


# ----------------------------------------------------#
#   模型实例化 和 部署
# ----------------------------------------------------#
n_features = 4  # 根据你的特征数量进行调整
output_size = 5
model = TimeSeriesClassifier(n_features=n_features, output_size=output_size)
model.load_state_dict(torch.load('./lstm_model.pth'))
model.to(device)

# 打印模型结构
logging.info(model)

'''
/****************************************************/
    训练过程
/****************************************************/
'''
# 设置训练参数
epochs = 200  # 训练轮数，根据需要进行调整
batch_size = 64  # 批大小，根据你的硬件调整

# DataLoader 加载数据集
# 将数据集转换为张量并创建数据加载器
# train_dataset = torch.utils.data.TensorDataset(train, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# validation_dataset = torch.utils.data.TensorDataset(validation, validation_target)
validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
val_num = len(test_dataset)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 学习率和优化策略
learning_rate = 1e-3
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # 设置学习率下降策略


# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
def calculate_accuracy(y_pred, y_true):
    # _, predicted_labels = torch.max(y_pred, 1)
    correct = torch.eq(y_pred, y_true).sum()
    accuracy = correct.sum() / len(correct)
    return accuracy

if False:
    model.eval()
    val_accuracy = 0
    acc = 0.0
    true_labels,pred_labels=[],[]
    with torch.no_grad():
        for inputs, labels in validation_loader:  # Assuming validation_loader is defined
            outputs = model(inputs.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            true_labels.extend(labels.tolist())
            pred_labels.extend(predict_y.tolist())    
    metrics = classification_report(true_labels, pred_labels)
    print(metrics)
    sys.exit(0)

if True:
    human_data_path = '/home/zhaihaonan/user_series/'
    sequences,labels = read_dataset(human_data_path,128)
    human_arr_tensor = torch.from_numpy(np.array(sequences))
    human_label_tensor = torch.from_numpy(np.array(labels))
    # 创建 TensorDataset
    human_dataset = TensorDataset(human_arr_tensor, human_label_tensor)
    human_loader = torch.utils.data.DataLoader(dataset=human_dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    human_output,bot_output=[],[]
    with torch.no_grad():
        for inputs, labels in human_loader:  # Assuming validation_loader is defined
            outputs = model(inputs.to(device))
            human_output.append(outputs.tolist())
    with torch.no_grad():
        for inputs, labels in validation_loader:  # Assuming validation_loader is defined
            outputs = model(inputs.to(device))
            bot_output.append(outputs.tolist())
    with open('user_output.txt','w',encoding='utf-8') as f:
        for i in human_output:
            f.write(str(i)+'\n')
    with open('bot_output.txt','w',encoding='utf-8') as f:
        for i in bot_output:
            f.write(str(i)+'\n')
    sys.exit(0)

best_acc = 0.0
for epoch in range(epochs):
    model.train()  # 将模型设置为训练模式
    train_epoch_loss = []
    train_epoch_accuracy = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # 获取输入数据和标签
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs.to(device))  # 前向传播
        predict_y = torch.max(outputs, dim=1)[1]
        loss = criterion(outputs, labels.to(device))
        loss.backward()  # 反向传播和优化
        optimizer.step()
        if i % 100 == 0:  # 每10个批次打印一次
            logging.info("--------------------------------------------")
            logging.info(f'Epoch {epoch + 1},step {i} Loss: {loss}')

    # Validation accuracy
    model.eval()
    val_accuracy = 0
    acc = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:  # Assuming validation_loader is defined
            outputs = model(inputs.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()
    val_accuracy = acc / val_num
    logging.info(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}')
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), './lstm_model.pth')
        logging.info('save best model!')
print('Finished Training')
