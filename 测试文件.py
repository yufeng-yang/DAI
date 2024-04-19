import pygame
import random
import sys
import time
from A_star import a_star_algorithm
from Deep_q_learning import *
import logging
import torch
import os

pygame.init()

# Basic parameters:
# cell_size: size of each cell
# grid_size: 12*12 grid
cell_size = 35
grid = [12, 12]
grid_size = 12
screen_size = grid_size * cell_size
score_height = 75
screen = pygame.display.set_mode((screen_size, screen_size + score_height))

# 颜色设置
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

# 为了实验的公平性，我决定使用可重复的随机生成的种子
# random.seed(random.randint(40, 50))
random.seed(40)
# # Used to judge performance
# score = 0
# steps = 0

# 初始化日志
pygame.display.set_caption('Deep Q-learning Snake')


def draw_grid():
    for x in range(0, screen_size, cell_size):
        for y in range(score_height, screen_size + score_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, WHITE, rect, 1)


# 生成食物并返回网格坐标
def place_food(grid_size, snake_body):
    while True:
        food_x = random.randint(0, grid_size - 1)
        food_y = random.randint(0, grid_size - 1)
        # 检查生成的食物位置是否与蛇身体的任何部分重叠
        if (food_x, food_y) not in snake_body:
            return food_x, food_y

# 在自动化算法时，算出蛇的朝向用的
def get_direction(current, next_position):
    if next_position[1] == current[1] - 1:
        return 'UP'
    elif next_position[1] == current[1] + 1:
        return 'DOWN'
    elif next_position[0] == current[0] - 1:
        return 'LEFT'
    elif next_position[0] == current[0] + 1:
        return 'RIGHT'
    else:
        return 'UP'

# 绘制蛇，包含蛇头和蛇身
def draw_snake(snake, direction):
    head_grid_x, head_grid_y = snake[0]
    head_x = head_grid_x * cell_size
    head_y = head_grid_y * cell_size + score_height
    arrow_length = cell_size // 2
    center_x = head_x + cell_size // 2
    center_y = head_y + cell_size // 2

    if direction == 'UP':
        points = [(center_x, head_y), (head_x, center_y), (head_x + cell_size, center_y)]
    elif direction == 'DOWN':
        points = [(center_x, head_y + cell_size), (head_x, center_y), (head_x + cell_size, center_y)]
    elif direction == 'LEFT':
        points = [(head_x, center_y), (center_x, head_y), (center_x, head_y + cell_size)]
    elif direction == 'RIGHT':
        points = [(head_x + cell_size, center_y), (center_x, head_y), (center_x, head_y + cell_size)]

    pygame.draw.polygon(screen, ORANGE, points)

    # 绘制蛇的其它部分
    for segment in snake[1:]:
        seg_x = segment[0] * cell_size
        seg_y = segment[1] * cell_size + score_height
        pygame.draw.rect(screen, GREEN, pygame.Rect(seg_x, seg_y, cell_size, cell_size))

    # print(snake)


# 绘制按钮的函数
def draw_button(screen, text, x, y, width, height, text_color, button_color, font):
    pygame.draw.rect(screen, button_color, (x, y, width, height))
    text_surf = font.render(text, True, text_color)
    screen.blit(text_surf, (x + (width - text_surf.get_width()) // 2, y + (height - text_surf.get_height()) // 2))


# 判断按钮是否被点击
def is_button_pressed(event, x, y, width, height):
    if event.type == pygame.MOUSEBUTTONDOWN:
        if x <= event.pos[0] <= x + width and y <= event.pos[1] <= y + height:
            return True
    return False


# 一键一格的更新位置
# 输入蛇头，返回新头的位置
def update_snake(snake_old_head, direction):
    head_x, head_y = snake_old_head[0]
    if direction == 'UP':
        head_y -= 1
    elif direction == 'DOWN':
        head_y += 1
    elif direction == 'LEFT':
        head_x -= 1
    elif direction == 'RIGHT':
        head_x += 1
    new_head = (head_x, head_y)
    return new_head


def check_collision(snake, grid_size):
    head_x, head_y = snake[0]
    # 检查蛇头是否出界或撞到自己
    if not (0 <= head_x < grid_size and 0 <= head_y < grid_size) or (head_x, head_y) in snake[1:]:
        return True
    return False


def handle_food_interaction(snake, food_position, grid_size):
    if snake[0] == food_position:
        # 蛇头和食物重合，蛇增长一格，不移除蛇尾
        food_position = place_food(grid_size, snake)
        # print(food_position)
    else:
        # 移除蛇尾
        snake.pop()
        # print(snake)
    return food_position

# 用于训练神经网络时，获得当前的状态文件
def get_current_state(snake, food_position, grid_size):
    snake_head = snake[0]
    direction_vector, distance = get_food_direction_and_distance(snake_head, food_position)
    body_distances = get_body_distances(snake, grid_size)

    # 将这些信息组合成一个状态向量
    state = direction_vector + (distance,) + tuple(body_distances)
    return state


# 所有需要训练的模型调用，绘制轮数用
def draw_training_rounds_text(surface, text, color, rect, font):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    surface.blit(text_surface, text_rect)


# 开始界面
def show_start_screen():
    button_width = 150
    button_height = 40
    spacing = 20
    column1_x = screen_size // 2 - button_width - 10
    column2_x = screen_size // 2 + 10
    # 设置字体大小
    font = pygame.font.Font(None, 24)

    buttons = [
        ("Human Player", column1_x, screen_size // 2 - (button_height + spacing)),
        ("A-Star Algorithm", column2_x, screen_size // 2 - (button_height + spacing)),
        ("Genetic Algorithm", column1_x, screen_size // 2),
        ("Q-Learning", column2_x, screen_size // 2),
        ("DQN-Train", column1_x, screen_size // 2 + (button_height + spacing)),
        ("DQN-Test", column2_x, screen_size // 2 + (button_height + spacing))
    ]

    running = True
    while running:
        screen.fill(BLACK)
        for text, x, y in buttons:
            draw_button(screen, text, x, y, button_width, button_height, WHITE, RED, font)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            for text, x, y in buttons:
                if is_button_pressed(event, x, y, button_width, button_height):
                    if text == "Human Player":
                        game_loop_human()
                    elif text == "A-Star Algorithm":
                        game_loop_a_star()
                    elif text == "Genetic Algorithm":
                        game_loop_genetic()
                    elif text == "Q-Learning":
                        game_loop_q_learning()
                    elif text == "CNN":
                        game_loop_cnn()
                    elif text == "DQN-Train":
                        game_loop_deep_q_learning_train()
                    elif text == "DQN-Tests":
                        game_loop_deep_q_learning_test()
                    running = False


        pygame.display.update()


def game_loop_human():
    running = True

    # Used to monitor whether the player fails
    game_over = False

    score = 0
    steps = 0

    # 定义，定义初始蛇放在循环之外，以免蛇的位置被无限刷新，但绘制时放在循环中的，和food一样
    # 首先定义蛇是一个list, [0]存储蛇头的位置
    snake = [(grid_size // 2, grid_size // 2)]

    # 生成食物，food_position存储了食物的网格位置
    food_position = place_food(grid_size, snake)

    snake_direction = 'UP'  # 蛇的初始方向

    # Location of Back Button
    back_button = (175, 0, 50, 30)
    font2 = pygame.font.Font(None, 24)

    # 创建一个Clock对象
    clock = pygame.time.Clock()
    # 设置帧率
    fps = 10


    while running:
        screen.fill(BLACK)  # 清空屏幕以准备新的绘制

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            # 检测按键，改变方向, 并防止反向移动
            if not game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and snake_direction != 'DOWN':
                        snake_direction = 'UP'
                    elif event.key == pygame.K_DOWN and snake_direction != 'UP':
                        snake_direction = 'DOWN'
                    elif event.key == pygame.K_LEFT and snake_direction != 'RIGHT':
                        snake_direction = 'LEFT'
                    elif event.key == pygame.K_RIGHT and snake_direction != 'LEFT':
                        snake_direction = 'RIGHT'

            # Back按钮的交互
            if event.type == pygame.MOUSEBUTTONDOWN:
                if back_button[0] <= event.pos[0] <= back_button[0] + back_button[2] and \
                   back_button[1] <= event.pos[1] <= back_button[1] + back_button[3]:
                    show_start_screen()
                    running = False

        # 检查碰撞，如果撞到了
        if check_collision(snake, grid_size):
            game_over = True
            draw_button(screen, 'Back', *back_button, WHITE, RED, font2)
            print(snake)

        # 更新蛇的位置
        if not game_over:
            new_head = update_snake(snake, snake_direction)
            snake.insert(0, new_head)
            steps += 1

        # 处理食物交互
        if not game_over:
            if snake[0] == food_position:
                score += 10
                food_position = handle_food_interaction(snake, food_position, grid_size)

            else:
                if len(snake) > 1:  # 确保蛇的长度大于1才执行pop
                    snake.pop()  # 没有吃到食物，移除尾部元素

        # 绘制网格
        draw_grid()

        # 绘制红色方块，作为食物，并占一格. 绘制的时候将网格坐标转换为像素，但食物还是以网格坐标储存
        pygame.draw.rect(screen, RED, (
            food_position[0] * cell_size, food_position[1] * cell_size + score_height, cell_size, cell_size))

        # 绘制初始蛇
        draw_snake(snake, snake_direction)

        # 绘制积分榜
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        steps_text = font.render(f"Steps: {steps}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(steps_text, (screen_size - 140, 10))

        pygame.display.flip()

        clock.tick(fps)



def game_loop_a_star():
    running = True
    path_found = True  # 增加一个标志位来标记是否找到路径
    snake = [(grid_size // 2, grid_size // 2)]  # 初始蛇的位置
    # 生成食物，food_position存储了食物的网格位置
    food_position = place_food(grid_size, snake)

    # Used to judge performance
    score = 0
    steps = 0

    path = []  # 存储从蛇头到食物的路径

    # Location of Game Over Button
    game_over_button = (175, 0, 50, 30)

    while running:
        screen.fill(BLACK)  # 清屏

        # 每当蛇头到达路径的终点或食物被吃掉，重新计算路径
        if not path or snake[0] == path[-1]:
            path = a_star_algorithm(grid_size, snake[0], food_position, snake)
            if not path:
                path_found = False  # 路径未找到，更新标志位

        # 移动蛇：将路径的下一个位置设为新的蛇头
        if path:
            print(snake)
            new_head = path.pop(0)
            direction = get_direction(snake[0], new_head)
            snake.insert(0, new_head)
            steps += 1  # 增加步数

            # 检查是否吃到食物
            if snake[0] == food_position:
                score += 10  # 增加分数
                # After the last food is eaten, food is generated again
                food_position = place_food(grid_size, snake)
                path = a_star_algorithm(grid_size, snake[0], food_position, snake)  # 重新计算路径
            else:
                snake.pop()  # 移除蛇尾，仅当没有吃到食物时

        # 绘制网格
        draw_grid()
        # 绘制蛇和食物
        draw_snake(snake, direction)
        pygame.draw.rect(screen, RED, (
        food_position[0] * cell_size, food_position[1] * cell_size + score_height, cell_size, cell_size))

        if not path_found:
            font = pygame.font.Font(None, 30)
            text_surf = font.render("I Can't Find Path", True, RED)
            text_rect = text_surf.get_rect(center=(screen_size // 2, score_height // 2 + 20))
            screen.blit(text_surf, text_rect)

            # 绘制 Game Over 按钮
            draw_button(screen, 'Back', *game_over_button, WHITE, RED, font)

        # 更新屏幕显示
        pygame.display.flip()

        # 按下Game Over
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_over_button[0] <= event.pos[0] <= game_over_button[0] + game_over_button[2] and \
                   game_over_button[1] <= event.pos[1] <= game_over_button[1] + game_over_button[3]:
                    show_start_screen()  # 返回到开始界面



        # 显示分数和步数
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        steps_text = font.render(f"Steps: {steps}", True, WHITE)
        screen.blit(score_text, (10, 10))  # 在左上角显示分数
        screen.blit(steps_text, (screen_size - 140, 10))  # 在右上角显示步数


        pygame.display.flip()  # 更新屏幕显示

        # 检查游戏结束条件等
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        time.sleep(0.02)  # 控制游戏速度

    pygame.quit()


def game_loop_genetic():
    print("Starting Genetic Algorithm game loop...")
    # 在这里实现使用遗传算法的游戏逻辑
    pass

def game_loop_q_learning():
    print("Starting Q-Learning game loop...")
    # 在这里实现使用Q-learning的游戏逻辑
    pass

def game_loop_cnn():
    print("Starting CNN game loop...")
    # 在这里实现使用卷积神经网络的游戏逻辑
    pass


def game_loop_deep_q_learning_test():
    print("Starting DNN game loop...")
    # 在这里实现使用卷积神经网络的游戏逻辑
    pass

def game_loop_deep_q_learning_train():
    # # 游戏初始化,测试时用这里
    # running = True
    # snake = [(grid_size // 2, grid_size // 2)]  # 初始蛇的位置
    # food_position = place_food(grid_size, snake)
    # direction = 'UP'  # 初始方向



    # 模型初始化
    input_size = 7  # 假设你的状态向量有7个维度
    output_size = 4  # 四个动作
    model = SnakeNet(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(3000)
    training_rounds = 0
    font = pygame.font.Font(None, 30)  # 选择合适的字号

    # 初始化奖励参数
    closest_breakthroughs = 0
    min_distance = float('inf')  # 设置为无限大，以便任何实际距离都会小于它

    while True:  # 外层循环，游戏重新开始时继续
        # 游戏初始化
        snake = [(grid_size // 2, grid_size // 2)]  # 初始蛇的位置
        # 在蛇头后面添加一格身体，假设初始方向向上，那么身体应该在蛇头的下方
        initial_body_part = (snake[0][0], snake[0][1] + 1)
        snake.append(initial_body_part)  # 添加身体部分
        food_position = place_food(grid_size, snake)
        current_min_distance = float('inf')
        running = True
        training_rounds += 1  # 每重新开始一次游戏，训练轮数增加
        closest_breakthroughs = 0  # 每次开始新游戏时重置


        # 每5000轮保存一下
        if training_rounds % 5000 == 0:
            filename = f"snake_model_{training_rounds}.pth"
            save_model(model, optimizer, filename)
            print(f"在训练轮次 {training_rounds} 保存模型到 {filename}")
            print(steps)
            print(score)


        # 统计一局内的步数和分数
        score = 0
        steps = 0


        while running:
            screen.fill(BLACK)

            # 绘制训练轮数
            draw_training_rounds_text(screen, f'Training Rounds: {training_rounds}', pygame.Color('white'), [0, 0, screen_size, score_height+30], font)

            # 首先调用get_current_state，得到得到食物与蛇头的相对方向和相对曼哈顿距离以及
            # 蛇头可移动的四个方向上，距离身体和边界的距离
            current_state = get_current_state(snake, food_position, grid_size)
            action_index = predict(model, current_state)  # 使用模型预测下一步动作
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            action = actions[action_index]
            steps += 1  # 增加步数

            # 测试区
            # print(current_state)
            # print("==============")
            # print(action_index)
            # print('***********')


            # 做出行动后更新蛇list
            new_head = update_snake(snake, action)
            next_state = get_current_state([new_head] + snake, food_position, grid_size)  # 更新后的状态

            # 计算新的距离
            new_distance = abs(new_head[0] - food_position[0]) + abs(new_head[1] - food_position[1])

            # 计算奖励和是否完成
            reward = 0
            done = False
            if new_head == food_position:
                reward = 10  # 吃到食物
                current_min_distance = float('inf')  # 重置最小距离
                closest_breakthroughs = 0  # 重置突破计数
            elif new_head in snake or not (0 <= new_head[0] < grid_size and 0 <= new_head[1] < grid_size):
                reward = -20  # 增大碰撞的惩罚
                done = True
            elif new_distance < current_min_distance:
                closest_breakthroughs += 1
                reward = closest_breakthroughs  # 根据突破次数增加奖励
                current_min_distance = new_distance

            replay_buffer.push(current_state, action_index, reward, next_state, done)
            train_model(model, optimizer, replay_buffer, batch_size=32)
            # print(reward)

            if done:
                # show_start_screen()  # 显示开始屏幕或重置提示
                running = False  # 结束内层循环，重新开始新一轮游戏

            snake.insert(0, new_head)  # 在列表前插入新的蛇头

            if new_head == food_position:
                score += 10  # 增加分数
                food_position = place_food(grid_size, snake)  # 重新放置食物
            else:
                snake.pop()  # 移除蛇尾

            # 绘制游戏元素
            draw_grid()
            draw_snake(snake, action)
            pygame.draw.rect(screen, RED, (
                food_position[0] * cell_size, food_position[1] * cell_size + score_height, cell_size, cell_size))

            # 显示分数和步数
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {score}", True, WHITE)
            steps_text = font.render(f"Steps: {steps}", True, WHITE)
            screen.blit(score_text, (10, 10))  # 在左上角显示分数
            screen.blit(steps_text, (screen_size - 140, 10))  # 在右上角显示步数

            # 更新屏幕显示
            pygame.display.flip()

            # time.sleep(0.1)  # 控制游戏速度

            # "'先用一下随机动作选择，看看能不能跑动代码'"
            # # 随机选择动作
            # action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            # print(snake)

            # replay_buffer.push(current_state, action, reward, next_state, done)
            # train_model(model, optimizer, replay_buffer, batch_size=32)
            #
            # # 更新蛇的位置
            # new_head = update_snake(snake, action)
            #
            # # 检查是否碰到自己或超出边界
            # if new_head in snake or not (0 <= new_head[0] < grid_size and 0 <= new_head[1] < grid_size):
            #     running = False
            #     show_start_screen()
            #     continue
            #
            # snake.insert(0, new_head)  # 在列表前插入新的蛇头
            #
            # if new_head == food_position:
            #     food_position = place_food(grid_size, snake)  # 重新放置食物
            # else:
            #     snake.pop()  # 移除蛇尾
            #
            # # 绘制游戏元素
            # draw_grid()
            # draw_snake(snake, action)
            # pygame.draw.rect(screen, RED, (
            # food_position[0] * cell_size, food_position[1] * cell_size + score_height, cell_size, cell_size))
            #
            # # 更新屏幕显示
            # pygame.display.flip()
            #
            # time.sleep(0.1)  # 控制游戏速度

    pygame.quit()



def calculate_reward(snake, food_position, new_head, current_min_distance):
    if new_head == food_position:
        return 10, False
    elif new_head in snake or not (0 <= new_head[0] < grid_size and 0 <= new_head[1] < grid_size):
        return -20, True
    else:
        new_distance = abs(new_head[0] - food_position[0]) + abs(new_head[1] - food_position[1])
        if new_distance < current_min_distance:
            return 1, False  # Reward for moving closer to the food
        return 0, False


def show_game_over_screen():
    running = True
    while running:
        screen.fill(BLACK)
        # # 显示得分和其他统计信息
        # messages = [
        #     f"Game Over!",
        #     f"Survival Time: {survival_time} seconds",
        #     f"Food Eaten: {food_count}",
        #     f"Score: {score}"
        # ]
        # for i, message in enumerate(messages):
        #     msg_surface = font.render(message, True, WHITE)
        #     screen.blit(msg_surface, (screen_size // 2 - msg_surface.get_width() // 2, 100 + i * 50))

        # draw_button(screen, 'Restart', *restart_button, WHITE, RED)
        # draw_button(screen, 'Main Menu', *main_menu_button, WHITE, RED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # if is_button_pressed(event, *restart_button):
            #     running = False
            #     game_loop()
            # if is_button_pressed(event, *main_menu_button):
            #     running = False
            #     show_start_screen()

        pygame.display.update()


show_start_screen()
