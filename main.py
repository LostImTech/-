import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import copy
import random


# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义客户端类
class Client:
    def __init__(self, client_id, malicious=False):
        self.client_id = client_id
        self.malicious = malicious
        self.train_loader, self.test_loader = self.load_data()
        ## 如下代码实现了各客户端本地模型的随机初始化；
        ## 在一个去中心化的场景中，随机初始化模型是合理的；
        ## 在一个中心化场景中，初始模型参数应该由聚合节点分配；
        self.local_model = SimpleNN()

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        random.seed(self.client_id)
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices, val_indices = indices[:split], indices[split:]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader

    def local_train(self, epochs=1, lr=0.01):
        # 模拟恶意节点的投毒攻击
        if self.malicious:
            poisoned_state_dict = {key: torch.rand_like(param) for key, param in self.local_model.state_dict().items()}
            self.local_model.load_state_dict(poisoned_state_dict)
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.local_model.parameters(), lr=lr)
            self.local_model.train()
            for epoch in range(epochs):
                for data, target in self.train_loader:
                    optimizer.zero_grad()
                    output = self.local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()


# 定义联邦学习类
class FederatedLearning:
    def __init__(self, num_clients, num_rounds, malicious_clients, selected_clients_num):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.malicious_clients = malicious_clients
        self.selected_clients_num = selected_clients_num
        self.clients = []
        # 初始化客户端
        self.init_clients()
        self.global_model = SimpleNN()
        self.test_losses_history = []
        self.test_accs_history = []

    def init_clients(self):
        malicious_client_ids = random.sample(range(self.num_clients), self.malicious_clients)
        for client_id in range(self.num_clients):
            malicious = client_id in malicious_client_ids
            client = Client(client_id, malicious)
            self.clients.append(client)

    def federated_learning(self):
        for round in range(self.num_rounds):
            local_models = []
            selected_clients = random.sample(self.clients, self.selected_clients_num)

            for client in selected_clients:
                client.local_model = copy.deepcopy(self.global_model)
                client.local_train()
                local_models.append(copy.deepcopy(client.local_model.state_dict()))

            # 参数聚合
            global_state_dict = self.global_model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.stack([local_models[i][key] for i in range(len(selected_clients))], 0).mean(0)

            self.global_model.load_state_dict(global_state_dict)

            # 测试全局模型
            test_loss_round, test_acc_round = self.evaluate_global_model()
            self.test_losses_history.append(test_loss_round)
            self.test_accs_history.append(test_acc_round)

            print(f'Round {round + 1}: Test Loss: {test_loss_round:.4f}, Test Accuracy: {test_acc_round:.4f}')

    def evaluate_global_model(self):
        self.global_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.clients[0].test_loader:  # 取第一个客户端的测试集进行评估
                output = self.global_model(data)
                loss = criterion(output, target)

                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss = running_loss / len(self.clients[0].test_loader.dataset)
        test_acc = correct / total

        return test_loss, test_acc


# 运行联邦学习
fl = FederatedLearning(num_clients=5, num_rounds=5, malicious_clients=0, selected_clients_num=5)
fl.federated_learning()

# 绘制全局模型的测试损失和准确率曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(fl.test_losses_history) + 1), fl.test_losses_history, marker='o')
plt.xlabel('Round')
plt.ylabel('Test Loss')
plt.title('Global Model Test Loss over Rounds')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(fl.test_accs_history) + 1), fl.test_accs_history, marker='o')
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.title('Global Model Test Accuracy over Rounds')

plt.tight_layout()
plt.show()

# 保存最终模型
torch.save(fl.global_model.state_dict(), "federated_model.pth")