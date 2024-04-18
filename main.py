import pygame
import random
import sys
import time
import A_star

pygame.init()

# Basic parameters:
# cell_size: size of each cell
# grid_size: 12*12 grid
cell_size = 35
grid_size = 12
screen_size = grid_size * cell_size
score_height = 50
screen = pygame.display.set_mode((screen_size, screen_size + score_height))
font = pygame.font.Font(None, 36)

# 颜色设置
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

def draw_grid():
    for x in range(0, screen_size, cell_size):
        for y in range(score_height, screen_size + score_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, WHITE, rect, 1)

def place_food(grid_size, cell_size):
    random.seed(42)
    food_x = random.randint(0, grid_size - 1) * cell_size
    food_y = random.randint(0, grid_size - 1) * cell_size + score_height
    return food_x, food_y

def draw_snake(snake, direction):
    head_x, head_y = snake[0]
    arrow_length = cell_size // 2
    center_x, center_y = head_x + cell_size // 2, head_y + cell_size // 2
    if direction == 'UP':
        points = [(center_x, head_y), (head_x, head_y + arrow_length), (head_x + cell_size, head_y + arrow_length)]
    elif direction == 'DOWN':
        points = [(center_x, head_y + cell_size), (head_x, head_y + cell_size - arrow_length), (head_x + cell_size, head_y + cell_size - arrow_length)]
    elif direction == 'LEFT':
        points = [(head_x, center_y), (head_x + arrow_length, head_y), (head_x + arrow_length, head_y + cell_size)]
    elif direction == 'RIGHT':
        points = [(head_x + cell_size, center_y), (head_x + cell_size - arrow_length, head_y), (head_x + cell_size - arrow_length, head_y + cell_size)]
    pygame.draw.polygon(screen, ORANGE, points)
    for segment in snake[1:]:
        pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], cell_size, cell_size))

def draw_button(screen, text, x, y, width, height, text_color, button_color):
    pygame.draw.rect(screen, button_color, (x, y, width, height))
    text_surf = font.render(text, True, text_color)
    screen.blit(text_surf, (x + (width - text_surf.get_width()) // 2, y + (height - text_surf.get_height()) // 2))

def is_button_pressed(event, x, y, width, height):
    if event.type == pygame.MOUSEBUTTONDOWN:
        if x <= event.pos[0] <= x + width and y <= event.pos[1] <= y + height:
            return True
    return False

def show_start_screen():
    start_button = (screen_size // 2 - 100, screen_size // 2 - 20, 200, 40)
    running = True
    while running:
        screen.fill(BLACK)
        draw_button(screen, 'Start', *start_button, WHITE, RED)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if is_button_pressed(event, *start_button):
                running = False
        pygame.display.update()
    game_loop()

def update_snake_direction_a_star(current_head, next_step):
    # 蛇头当前位置
    current_x, current_y = current_head
    # 下一步位置
    next_x, next_y = next_step

    # 确定下一步的方向
    if next_x == current_x and next_y == current_y - cell_size:
        new_direction = 'UP'
    elif next_x == current_x and next_y == current_y + cell_size:
        new_direction = 'DOWN'
    elif next_x == current_x - cell_size and next_y == current_y:
        new_direction = 'LEFT'
    elif next_x == current_x + cell_size and next_y == current_y:
        new_direction = 'RIGHT'
    else:
        new_direction = None  # 无效移动（不应发生）

    return new_direction


def game_loop():
    start_time = time.time()  # 记录游戏开始时间
    clock = pygame.time.Clock()
    fps = 5
    score = 0

    # 假设蛇初始位置在屏幕中央
    snake = [(screen_size // 2, screen_size // 2 + score_height)]
    food_position = place_food(grid_size, cell_size)
    snake_direction = 'UP'

    # 用来储存从A*返回的路径
    path = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        if not path:
            # 每当路径用完时，重新计算路径
            path = A_star.astar(snake,
                                [(food_position[0] // cell_size, (food_position[1] - score_height) // cell_size)],
                                grid_size, grid_size)

        if path and len(path) > 1:
            next_step = path.pop(0)
            snake_direction = update_snake_direction_a_star(snake[0], (
            next_step[0] * cell_size, next_step[1] * cell_size + score_height))

        head_x, head_y = snake[0]
        if snake_direction == 'UP':
            head_y -= cell_size
        elif snake_direction == 'DOWN':
            head_y += cell_size
        elif snake_direction == 'LEFT':
            head_x -= cell_size
        elif snake_direction == 'RIGHT':
            head_x += cell_size

        new_head = (head_x, head_y)

        if new_head in snake or new_head[0] < 0 or new_head[0] >= screen_size or new_head[1] < score_height or new_head[
            1] >= screen_size + score_height:
            show_game_over_screen(start_time, score)
            break

        snake.insert(0, new_head)

        if new_head == food_position:
            score += 10
            food_position = place_food(grid_size, cell_size)
            path = []  # 清空路径，因为食物位置变了
        else:
            snake.pop()

        screen.fill(BLACK)
        draw_grid()
        score_text = font.render(f"Score: {score}", True, RED)
        screen.blit(score_text, (5, 10))
        pygame.draw.rect(screen, RED, (food_position[0], food_position[1], cell_size, cell_size))
        draw_snake(snake, snake_direction)
        pygame.display.update()

        clock.tick(fps)


def show_game_over_screen(start_time, score):
    end_time = time.time()
    survival_time = format(end_time - start_time, ".2f")  # 计算生存时间（秒）
    food_count = score // 10  # 每10分代表吃掉一个食物

    restart_button = (screen_size // 2 - 100, screen_size // 2 + 100, 200, 40)
    main_menu_button = (screen_size // 2 - 100, screen_size // 2 + 150, 200, 40)

    running = True
    while running:
        screen.fill(BLACK)
        # 显示得分和其他统计信息
        messages = [
            f"Game Over!",
            f"Survival Time: {survival_time} seconds",
            f"Food Eaten: {food_count}",
            f"Score: {score}"
        ]
        for i, message in enumerate(messages):
            msg_surface = font.render(message, True, WHITE)
            screen.blit(msg_surface, (screen_size // 2 - msg_surface.get_width() // 2, 100 + i * 50))

        draw_button(screen, 'Restart', *restart_button, WHITE, RED)
        draw_button(screen, 'Main Menu', *main_menu_button, WHITE, RED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if is_button_pressed(event, *restart_button):
                running = False
                game_loop()
            if is_button_pressed(event, *main_menu_button):
                running = False
                show_start_screen()

        pygame.display.update()

# 主循环入口
show_start_screen()