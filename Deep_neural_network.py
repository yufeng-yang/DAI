import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


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
