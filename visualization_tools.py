# Basic parameters:
# cell_size: size of each cell
# grid_size: 12*12 grid
import random

import pygame
Game_name = "DIA_Snake"
cell_size = 35
grid = [12, 12]
grid_size = 12
grid_num = grid_size * grid_size
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

# 用于 e-greedy
action_rng = random.Random()

# 初始化
pygame.display.set_caption('Welcome_to_Snake')

# 绘制网格
def draw_grid():
    for x in range(0, screen_size, cell_size):
        for y in range(score_height, screen_size + score_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, WHITE, rect, 1)


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
    # 只有一个头的时候
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
    else:
        points = "UP"

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

