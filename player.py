# def game_loop():
#     start_time = time.time()  # 记录游戏开始时间
#     clock = pygame.time.Clock()
#     fps = 5
#     score = 0
#     grid = DFS.Grid(grid_size, grid_size + score_height // cell_size)  # 初始化网格
#
#     food_position = place_food(grid_size, cell_size)
#     snake = [(screen_size // 2, screen_size // 2 + score_height)]
#     snake_direction = 'UP'
#     directions = {
#         pygame.K_UP: 'UP',
#         pygame.K_DOWN: 'DOWN',
#         pygame.K_LEFT: 'LEFT',
#         pygame.K_RIGHT: 'RIGHT'
#     }
#     opposite_directions = {
#         'UP': 'DOWN',
#         'DOWN': 'UP',
#         'LEFT': 'RIGHT',
#         'RIGHT': 'LEFT'
#     }
#
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 pygame.quit()
#                 sys.exit()
#             elif event.type == pygame.KEYDOWN:
#                 if event.key in directions and snake_direction != opposite_directions[directions[event.key]]:
#                     snake_direction = directions[event.key]
#         path = DFS.dfs(grid, snake[0], food_position)
#         if path and len(path) > 1:
#             next_step = path[1]
#             snake_direction = update_snake_direction_DFS(snake[0], next_step)
#             # print(11)
#
#         head_x, head_y = snake[0]
#         if snake_direction == 'UP':
#             head_y -= cell_size
#         elif snake_direction == 'DOWN':
#             head_y += cell_size
#         elif snake_direction == 'LEFT':
#             head_x -= cell_size
#         elif snake_direction == 'RIGHT':
#             head_x += cell_size
#
#         new_head = (head_x, head_y)
#
#         # 检查碰撞：边界和自身碰撞
#         if new_head in snake or new_head[0] < 0 or new_head[0] >= screen_size or new_head[1] < score_height or new_head[1] >= screen_size + score_height:
#             show_game_over_screen(start_time, score)  # 注意这里传递了 start_time 和 score
#             break  # 使用 break 而非 continue，因为不需要继续执行循环的后续部分
#
#         snake.insert(0, new_head)  # 更新蛇的头部位置
#
#         if new_head == food_position:
#             score += 10
#             food_position = place_food(grid_size, cell_size)
#         else:
#             snake.pop()
#
#         screen.fill(BLACK)
#         draw_grid()
#         score_text = font.render(f"Score: {score}", True, RED)
#         screen.blit(score_text, (5, 10))
#         pygame.draw.rect(screen, RED, (food_position[0], food_position[1], cell_size, cell_size))
#         draw_snake(snake, snake_direction)
#         pygame.display.update()
#
#         clock.tick(fps)