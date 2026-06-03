import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 模拟火星温度数据（昼夜周期+随机波动）
def generate_mars_temperature(days=365, noise_scale=0.2):
    t = np.linspace(0, days*24, days*24)  # 每小时一个数据点
    daily_cycle = -30 * np.cos(2 * np.pi * t / 24)  # 昼夜温差（-60°C ~ 0°C）
    seasonal_variation = 10 * np.sin(2 * np.pi * t / (24*30))  # 火星季节变化
    noise = noise_scale * np.random.normal(scale=10, size=len(t))
    temperature = daily_cycle + seasonal_variation + noise
    return temperature.reshape(-1, 1)  # 转换为(n_samples, 1)

# 数据标准化
data = generate_mars_temperature(days=60)  # 生成60天数据
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

def create_sequences(data, seq_length=24, pred_horizon=24):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_horizon])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# 参数设置
SEQ_LENGTH = 72  # 输入序列长度（3天）
PRED_HORIZON = 24  # 预测未来24小时

# 划分训练/测试集（80%-20%）
train_size = int(0.8 * len(data_normalized))
X, y = create_sequences(data_normalized, SEQ_LENGTH, PRED_HORIZON)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 转换为PyTorch DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
# 输出示例形状
print(f"Sample X shape: {X_train[0].shape}, y shape: {y_train[0].shape}")

class MarsTemperatureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        
        # 仅用最后一个时间步的隐藏状态预测未来序列
        last_hidden = lstm_out[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions.unsqueeze(-1)  # (batch_size, pred_horizon, 1)

# 初始化模型
model = MarsTemperatureLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=PRED_HORIZON
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

def train_model(model, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate(model)
        scheduler.step(val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

def evaluate(model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 开始训练
train_model(model, epochs=100)

def get_predictions(model, sample_idx=0):
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = X_test[sample_idx], y_test[sample_idx]
        pred = model(sample_x.unsqueeze(0))
        
    # 反标准化
    true = scaler.inverse_transform(sample_y.numpy())
    pred = scaler.inverse_transform(pred.squeeze(0).numpy())

get_predictions(model, sample_idx=10)
