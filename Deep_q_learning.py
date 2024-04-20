import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import os


# 神经网络结构
class SnakeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.out = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.out(x)
        return x


# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 日志文件的生成有关代码
class Trainer:
    def __init__(self, model, optimizer, log_dir='runs'):
        self.model = model
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def train_model(self, replay_buffer, batch_size):
        # 缓存中的数据不足以形成一个完整的批次，不执行训练
        if len(replay_buffer) < batch_size:
            return

        transitions = replay_buffer.sample(batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(batch[0], dtype=torch.float32)
        action_batch = torch.tensor(batch[1], dtype=torch.long)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32)
        done_batch = torch.tensor(batch[4], dtype=torch.float32)

        # 计算当前状态下模型预测的Q值
        q_values = self.model(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 计算下一个状态的最大预测Q值
        next_q_values = self.model(next_state_batch).max(1)[0]
        # 对于游戏结束的状态，我们将它的Q值设为0
        next_q_values[done_batch.bool()] = 0.0

        # 计算预期Q值
        expected_q_values = (next_q_values * 0.99) + reward_batch
        loss = torch.nn.functional.mse_loss(state_action_values, expected_q_values)

        # 计算损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 添加日志
        self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
        self.global_step += 1  # 每次训练迭代后递增global_step

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("模型成功从", filename, "加载")

    def close(self):
        self.writer.close()


# 得到食物与蛇头的相对方向和相对曼哈顿距离
def get_food_direction_and_distance(snake_head, food_position):
    head_x, head_y = snake_head
    food_x, food_y = food_position
    direction_vector = (food_x - head_x, food_y - head_y)
    distance = abs(direction_vector[0]) + abs(direction_vector[1])  # 曼哈顿距离
    return direction_vector, distance


# 蛇头可移动的四个方向上，距离身体和边界的距离
def get_body_distances(snake, grid_size):
    head_x, head_y = snake[0]
    distances = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    for dx, dy in directions:
        distance = 0
        while True:
            distance += 1
            nx, ny = head_x + dx * distance, head_y + dy * distance
            if (nx, ny) in snake[1:] or not (0 <= nx < grid_size and 0 <= ny < grid_size):
                break
        distances.append(distance)
    return distances


# 关于在dnn中，根据输出的方向更新蛇
def update_snake_dnn(snake_head, direction):
    x, y = snake_head
    if direction == 'UP':
        y -= 1
    elif direction == 'DOWN':
        y += 1
    elif direction == 'LEFT':
        x -= 1
    elif direction == 'RIGHT':
        x += 1
    return (x, y)


# 模型预测函数
def predict(model, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 转为tensor并增加batch维度
    with torch.no_grad():
        action_probs = model(state_tensor)
        action = action_probs.argmax().item()
    return action


def save_model(model, optimizer, filename="snake_model.pth"):
    """将模型和优化器状态保存到磁盘。"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)

def load_model(model, optimizer, filename="snake_model.pth"):
    """从磁盘加载模型和优化器状态。"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("模型成功从", filename, "加载")
    else:
        print("未找到模型文件", filename)
