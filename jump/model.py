import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_df = pd.read_csv('train_predict.csv')
train_data_df = train_data_df.iloc[:, 1:train_data_df.shape[1]]
# 滑动窗口逻辑
SEQ_LENGTH = 15  # 完整序列长度应当为15

sequences = []
targets = []
for i in tqdm(range(0, train_data_df.shape[0] - (SEQ_LENGTH - 1))):
    start_point = train_data_df.iloc[i]
    end_point = train_data_df.iloc[i + SEQ_LENGTH - 1]
    if start_point['traj_id'] != end_point['traj_id']:
        continue
    sequence = []
    for j in range(0, 14):
        sequence.append(train_data_df.iloc[i + j].values)
    sequences.append(torch.tensor(sequence).float())  # 前十四个点的特征
    res = []
    res.append(train_data_df.iloc[i + 14]['x'])
    res.append(train_data_df.iloc[i + 14]['y'])
    targets.append(torch.tensor(res).float())
train_seqs, test_seqs, train_tgts, test_tgts = train_test_split(sequences, targets, test_size=0.2, random_state=42)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))
        self.batch_norm = nn.BatchNorm1d(input_size)  # 加入归一化层

    def forward(self, input_seq):
        input_seq = self.batch_norm(input_seq)  # 归一化
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# EarlyStopping类的实现
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# 初始化EarlyStopping对象
early_stopping = EarlyStopping(patience=3, verbose=True)

# 确定输入输出大小
input_size = 8  # 包括空间信息和其他附加特征的总特征数量
output_size = 2  # 经度和纬度

model = LSTMModel(input_size=input_size, hidden_layer_size=100, output_size=output_size, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 初始化模型

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-5, verbose=True)
# 定义损失函数和优化器
criterion = nn.MSELoss()

# 训练模型
epochs = 5
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    print('start_train')
    for i in tqdm(range(len(train_seqs))):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                             torch.zeros(1, 1, model.hidden_layer_size, device=device))

        input_seq = train_seqs[i].to(device)
        target = train_tgts[i].to(device)
        y_pred = model(input_seq)

        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    # 输出训练损失
    print(f'Epoch {epoch} Loss: {running_loss / len(train_seqs)}')

    # 验证步骤
    model.eval()
    valid_loss = 0.0
    print('start_eval')
    with torch.no_grad():
        for i in range(len(test_seqs)):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                                 torch.zeros(1, 1, model.hidden_layer_size, device=device))

            input_seq = test_seqs[i].to(device)
            target = test_tgts[i].to(device)
            y_pred = model(input_seq)

            loss = criterion(y_pred, target)
            valid_loss += loss.item()

    # 输出验证损失
    print(f'Validation Loss: {valid_loss / len(test_seqs)}')

    # 调用EarlyStopping
    early_stopping(valid_loss / len(test_seqs), model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    scheduler.step(valid_loss / len(test_seqs))

# 保存训练好的模型
torch.save(model.state_dict(), 'model1.pth')