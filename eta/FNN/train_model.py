import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

DATASET_PATH = '../train.csv'  # 完整的74万条数据
MINI_DATASET_PATH = './mini_train.csv'  # 74万条数据的前一万条数据


def transfer_data(data):
    X = data.loc[:, ['holidays', 'time_period', 'cpath', 'cost', 'x1', 'y1', 'x2', 'y2', 'speed']]
    X = X.values
    y = data.loc[:, ['road_cost_time']]
    y = y.values
    return X, y

data = pd.read_csv(DATASET_PATH)
X, y = transfer_data(data=data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 随机抽选训练组与测试组

# 转换为PyTorch的张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
# 定义模型参数
INPUT_SIZE = X.shape[1]  # 输入特征的维度
HIDDEN_SIZES = [10, 20, 10]  # 隐藏层大小列表
OUTPUT_SIZE = 1  # 输出维度
lr = 0.005  # 学习率

# 构建前馈神经网络模型
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNN, self).__init__()
        layers = []
        input_dim = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 添加批归一化层
            layers.append(nn.ReLU())  # 或者其他的激活函数
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))  # 输出层
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # 实例化模型
    model = FeedForwardNN(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
        
    torch.save(model.state_dict(), 'model.pth')