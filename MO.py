import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from membership_inference_attacks import black_box_benchmarks
from privacy_risk_score_utils import calculate_risk_score
from privacy_risk_score_utils import distrs_compute

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 设置随机种子
torch.manual_seed(42)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# 初始化模型和优化器
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 初始化隐私引擎（Opacus v1.5.3）
privacy_engine = PrivacyEngine(
    accountant="rdp",  # 使用 RDP 会计师
    secure_mode=False
)

# 使训练过程私有
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)


# 训练循环（Opacus v1.5.3）
def train(model, train_loader, optimizer, criterion, epochs, privacy_engine):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        # 获取隐私预算（新版本）
        epsilon = privacy_engine.get_epsilon(delta=1e-5)

        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, '
              f'Accuracy: {100. * correct / total:.2f}%, '
              f'(ε = {epsilon:.2f}, δ = 1e-5)')


# 执行训练
train(model, train_loader, optimizer, criterion, epochs=5, privacy_engine=privacy_engine)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# 定义函数获取模型预测结果
def get_predictions(model, dataloader):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            all_outputs.append(output.numpy())
            all_labels.append(target.numpy())
    outputs = np.concatenate(all_outputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return outputs, labels

# 获取目标模型的训练集和测试集预测结果
target_train_outputs, target_train_labels = get_predictions(model, train_loader)
target_test_outputs, target_test_labels = get_predictions(model, test_loader)

# 训练影子模型（结构与目标模型相同）
shadow_model = Net()
shadow_optimizer = optim.SGD(shadow_model.parameters(), lr=0.05, momentum=0.9)

# 划分影子模型的训练集和测试集（示例使用MNIST训练集的前半部分作为影子训练，后半部分作为影子测试）
shadow_train_size = 30000  # MNIST训练集共60000样本
shadow_indices_train = list(range(0, shadow_train_size))
shadow_indices_test = list(range(shadow_train_size, 60000))

shadow_train_dataset = torch.utils.data.Subset(train_dataset, shadow_indices_train)
shadow_test_dataset = torch.utils.data.Subset(train_dataset, shadow_indices_test)

shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=1024, shuffle=True)
shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=1024, shuffle=False)
# 初始化影子模型和优化器
shadow_model = ShadowNet()
shadow_optimizer = optim.SGD(shadow_model.parameters(), lr=0.05, momentum=0.9)
shadow_criterion = nn.CrossEntropyLoss()


def train_shadow_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        print(f'Shadow Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, '
              f'Accuracy: {100. * correct / total:.2f}%')


# 执行训练
train_shadow_model(shadow_model, shadow_train_loader, shadow_optimizer, shadow_criterion, epochs=5)
def get_predictions(model, dataloader):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            all_outputs.append(output.numpy())
            all_labels.append(target.numpy())
    outputs = np.concatenate(all_outputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return outputs, labels

# 获取影子模型的预测结果
shadow_train_outputs, shadow_train_labels = get_predictions(shadow_model, shadow_train_loader)
shadow_test_outputs, shadow_test_labels = get_predictions(shadow_model, shadow_test_loader)
# 计算训练集准确率
shadow_train_preds = np.argmax(shadow_train_outputs, axis=1)
shadow_train_acc = np.mean(shadow_train_preds == shadow_train_labels)
print(f"Shadow Model Train Accuracy: {shadow_train_acc*100:.2f}%")

# 计算测试集准确率
shadow_test_preds = np.argmax(shadow_test_outputs, axis=1)
shadow_test_acc = np.mean(shadow_test_preds == shadow_test_labels)
print(f"Shadow Model Test Accuracy: {shadow_test_acc*100:.2f}%")

# 初始化攻击类
benchmarks = black_box_benchmarks(
    shadow_train_performance=(shadow_train_outputs, shadow_train_labels),
    shadow_test_performance=(shadow_test_outputs, shadow_test_labels),
    target_train_performance=(target_train_outputs, target_train_labels),
    target_test_performance=(target_test_outputs, target_test_labels),
    num_classes=1
)

# 执行攻击
benchmarks._mem_inf_benchmarks()

shadow_train_conf = np.array([shadow_train_outputs[i, shadow_train_labels[i]] for i in range(len(shadow_train_labels))])
shadow_test_conf = np.array([shadow_test_outputs[i, shadow_test_labels[i]] for i in range(len(shadow_test_labels))])
tr_values = shadow_train_conf  # 影子模型训练集置信度
te_values = shadow_test_conf    # 影子模型测试集置信度
tr_labels = shadow_train_labels  # 影子模型训练集标签
te_labels = shadow_test_labels   # 影子模型测试集标签

# 目标模型训练集的数据（需要评估风险）
target_train_conf = np.array([target_train_outputs[i, target_train_labels[i]] for i in range(len(target_train_labels))])
data_values = target_train_conf  # 目标模型训练集置信度
data_labels = target_train_labels  # 目标模型训练集标签

# 计算隐私风险评分
risk_scores = calculate_risk_score(
    tr_values=tr_values,
    te_values=te_values,
    tr_labels=tr_labels,
    te_labels=te_labels,
    data_values=data_values,
    data_labels=data_labels,
    num_bins=5,     # 分箱数量（可调整）
    log_bins=True    # 是否使用对数分箱（推荐True）
)
# 输出统计信息
print(f"risk_scores: {np.mean(risk_scores):.3f}")
