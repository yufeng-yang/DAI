import random
import argparse
import time

import numpy as np
# from ai_game import Game
# from settings import *
import os
import pygame as pg
import random
# from settings import *
import numpy as np
from GeneticAlgorithm.GA_Neural_Networks import *
import os
from visualization_tools import *


# 计算所需权重和偏置的总数，作为基因链的长度，也就是权重的数量
GENES_LEN = imput_size * layer1_size + layer1_size * layer2_size + layer2_size * layer3_size + layer3_size * Out_size + layer1_size + layer2_size + layer3_size + Out_size
# 父母代的规模，即一开始有100个基因. 并且后续进化中每次筛选，只选前100个基因组交配
Number_of_chosen_parents = 100

# 即那被选中的100个父母类，我想要让他们交配出的孩子的数量
Next_generation_number = 400

# 可供小蛇选择移动的方向（与检测方向不同）
MOVEABLE_DIRECTION = [(0, -1), (0, 1), (-1, 0), (1, 0)]
MUTATE_RATE = 0.1


class Snake:
    # 此处初始化小蛇
    def __init__(self, head_position, direction, genes, grid_length, grid_height):
        self.body = [head_position]
        self.direction = direction
        self.score = 0
        self.steps = 0
        self.dead = False
        self.useless_steps_set = [0] * grid_length * grid_height
        self.grid_length = grid_length
        self.grid_height = grid_height

        # Net 用于实例化生成神经网络(根据基因)
        self.Genetic_algorithm = Genetic_Algorithm_Network(imput_size, layer1_size, layer2_size, layer3_size, Out_size, genes.copy())
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 再找到神经网络的输入，即state
    # 有snake对象自己调用该方法，因此自带snake的一部分信息
    # 该函数只需要输入food的位置，就可以自动返回一个state，里面包含蛇头方向，蛇尾方向以及8个方向上距离食物和两种障碍的距离
    def get_state(self, food):
        # head_position direction.
        i = MOVEABLE_DIRECTION.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        # Tail direction.
        if len(self.body) == 1:
            tail_direction = self.direction
        else:
            tail_direction = (self.body[-2][0] - self.body[-1][0], self.body[-2][1] - self.body[-1][1])
        i = MOVEABLE_DIRECTION.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0
        state = head_dir + tail_dir

        # 相比于之前的四方向探索，这次增加为8方向
        seeable_directions = [[0, -1], [1, -1], [1, 0], [1, 1],
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]

        # 探知函数 用于探测蛇头8个方向距离body和墙和食物的距离
        for dir in seeable_directions:
            x = self.body[0][0] + dir[0]
            y = self.body[0][1] + dir[1]
            dis_to_food_or_body = 1.0
            see_food = 0.0
            see_self = 0.0
            while x >= 0 and x < self.grid_length and y >= 0 and y < self.grid_height:
                if (x, y) == food:
                    see_food = 1.0
                elif (x, y) in self.body:
                    see_self = 1.0
                dis_to_food_or_body += 1
                x += dir[0]
                y += dir[1]
            state += [1.0 / dis_to_food_or_body, see_food, see_self]

        return state

    # 用于移动小蛇的方法，由小蛇直接自己调用
    # 输入食物的位置，返回小蛇有没有吃掉食物
    # 位置移动已经完成，无需返回
    "'这是比较重要的一个方法，基于之前get_state中得到的，4（头方向）+4（尾方向）+3*8（8个方向上，离食物或障碍、蛇身的距离）共32的输入维度，这也是神经网络第一层的输入维度了'"
    def move(self, food):
        # 每走一步就 +1
        self.steps += 1
        # 这里就是刚刚定义好的上一个方法：get_state
        state = self.get_state(food)

        # 核心就在这了: 用训练好的神经网络，输入state, 从而预测行动
        action = self.Genetic_algorithm.predict(state)

        # 行动后更新蛇头的方向和位置
        self.direction = MOVEABLE_DIRECTION[action]
        head_position = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])

        has_eat = False

        # 判断小蛇是否死亡（触碰自己或者墙壁）
        if (head_position[0] < 0 or head_position[0] >= self.grid_length or head_position[1] < 0 or head_position[1] >= self.grid_height
                or head_position in self.body[:-1]):
            self.dead = True
        else:
            # 没死的话，就插入新蛇头
            self.body.insert(0, head_position)
            # 如果吃到食物
            if head_position == food:
                self.score += 10
                has_eat = True
            # 没吃到也没死，就pop旧的蛇尾
            else:
                self.body.pop()
                # 设定一个固定容量的useless_steps_set，控制当某一个食物位置时的移动，如果重复出现至超过一定容量，则判断进入死循环
                if (head_position, food) not in self.useless_steps_set:
                    self.useless_steps_set.append((head_position, food))
                    del self.useless_steps_set[0]
                # 判断已经进入死循环，直接算死亡
                else:
                    self.dead = True

        return has_eat


class Game:
    def __init__(self, genes_list, seed=None, show=False, rows=12, cols=12):
        # 默认大小还是12*12，但是可以通过传入来改。
        self.Y = rows
        self.X = cols
        self.show = show
        # 设置seed，从而复现训练，设在这里比全局定死避免过拟合
        self.seed = seed if seed is not None else random.randint(-99999999, 99999999)
        self.rand = random.Random(self.seed)

        # 多代理训练，根据基因列表，生成复数的蛇，加快速度
        # snakes 蛇集
        self.snakes = []
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        for genes in genes_list:
            head_position = self.rand.choice(board)
            direction = MOVEABLE_DIRECTION[self.rand.randint(0, 3)]
            self.snakes.append(Snake(head_position, direction, genes, self.X, self.Y))

        # 生成食物
        self.food = self.placeGA_food()
        self.best_score = 0

        if show:
            pg.init()
            self.width = cols * cell_size
            self.height = rows * cell_size + score_height

            pg.display.set_caption(Game_name)
            self.screen = pg.display.set_mode((self.width, self.height))
            self.clock = pg.time.Clock()

    # play函数通过调用move（move会调用GA网络进行预测），这里play函数通过调用move实现游戏
    # 无需输入，输出所有参与训练的蛇的分数，步数和本剧游戏的随机seed
    # 将会执行，直至所有的蛇死亡
    def play(self):
        # 开局当然所有的蛇都还可以自由活动
        moveable_snakes = set(self.snakes)
        score = []
        steps = []
        steps_for_visualisation = 0
        # 当还有蛇或者或者食物未被吃完，就是一局游戏还能进行下去的时候
        while moveable_snakes and self.food is not None:
            if self.show:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        return
                self.visualization()

            for snake in moveable_snakes:
                # 所有的蛇依次调用move，返回是否吃到食物
                has_eat = snake.move(self.food)
                steps_for_visualisation += 1
                if has_eat:
                    # 吃到了放个新的
                    self.food = self.placeGA_food()
                    # 万一所有食物都被吃完了，直接退出
                    if self.food is None:
                        print("Win")
                        break
                # 更新best_score
                if snake.score > self.best_score:
                    self.best_score = snake.score




            # 只留下活动的蛇
            # 代所有蛇都移动过后，统计一下剩下的蛇，然后把还可以行动的装回去
            left_snakes = []
            for snake in moveable_snakes:
                if not snake.dead:
                    left_snakes.append(snake)
            moveable_snakes = set(left_snakes)

        # 统计一局游戏下来，所有的蛇的得分情况
        for snake in self.snakes:
            score.append(snake.score)
            steps.append(snake.steps)
        # 仅有一只蛇参与训练
        if len(self.snakes) <= 1:
            score, steps = self.snakes[0].score, self.snakes[0].steps

        return score, steps, self.seed

    # 再次定义一下placeGA_food
    def placeGA_food(self):
        board = set()
        # 创建网格的坐标集合
        for x in range(self.X):
            for y in range(self.Y):
                board.add((x, y))

        # 避免与蛇重合
        for snake in self.snakes:
            if not snake.dead:
                for body in snake.body:
                    board.discard(body)

        # 没地方了的话，就不生成
        if len(board) == 0:
            return None

        # 随机生成
        # print(self.rand.choice(list(board)))
        # print("**************")
        return self.rand.choice(list(board))

    # 由于这里的direction是由元组表述的，因此需要多写一个。
    def direction_from_vector(self,vector):
        x, y = vector
        if x == 1 and y == 0:
            return 'RIGHT'
        elif x == -1 and y == 0:
            return 'LEFT'
        elif x == 0 and y == 1:
            return 'DOWN'
        elif x == 0 and y == -1:
            return 'UP'
        return 'UP'

    # 可视化游戏，通过调用visualization_tools
    def visualization(self):
        self.screen.fill(BLACK)  # 设定背景为黑色

        # 绘制网格
        draw_grid()

        # for snake in self.snakes:
        #     print(snake.direction)
        for snake in self.snakes:
            if not snake.dead:
                direction = self.direction_from_vector(snake.direction)
                draw_snake(snake.body, direction)

        # 绘制食物
        food_x, food_y = self.food
        food_rect = pg.Rect(food_x * cell_size, food_y * cell_size + score_height, cell_size, cell_size)
        pg.draw.rect(self.screen, RED, food_rect)

        # 显示分数和步数
        font = pg.font.Font(None, 36)
        score_text = font.render(f"Score: {self.best_score}", True, WHITE)
        steps_text = font.render(f"Steps: {self.snakes[0].steps}", True, WHITE)
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(steps_text, (self.width - 140, 5))

        # 更新
        pg.display.flip()
        # time.sleep(0.05)


# 定义一个游戏中的个体（说是个体，实际就是基因组作为权重的神经网络）
class IntelligentAgent:

    # 初始化每个个体需要的属性
    # genes: 一个个体的基因组，score：一个个体的分数，steps，一个个体的步数，fitness，一个个体的fitness，seed，复现用的种子
    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.fitness = 0
        self.seed = None

    # 得到每个个体的fitness，通过score和steps计算得到
    # 那哪来的score和steps呢？就是之前在ai_game中定义的game方法里得到的，game就是之前那个可以玩一盘游戏的方法
    # 传入基因组，调用Game开启游戏，获得steps和score最终计算fitness
    def get_fitness(self):
        game = Game([self.genes])
        self.score, self.steps, self.seed = game.play()
        self.fitness = (self.score + 1 / self.steps) * 100000


class PopulationEvolution:
    def __init__(self, total_number_of_parents=Number_of_chosen_parents, children_scale=Next_generation_number, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE):
        # 父母代的规模，即一开始有100个基因. 并且后续进化中每次筛选，只选前100个基因组交配
        self.total_number_of_parents = total_number_of_parents
        # 即那被选中的100个父母类，我想要让他们交配出的孩子的数量
        self.children_scale = children_scale
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        # fitness最高的小蛇
        self.best_individual = None
        self.average_score_of_all_agents = 0

    # 随机生成最初的祖先，思路就是-1到1之间随机生成基因
    def randomFirstAncestor(self):
        for i in range(self.total_number_of_parents):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(IntelligentAgent(genes))

    def inherit_ancestor(self):
        for i in range(self.total_number_of_parents):
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                self.population.append(IntelligentAgent(genes))

    """这里可以说是基因算法的工具，即根据以下四个工具才能得到最后的def evolve(self):。先从四个工具开始
    分别为def mutate(self, chosen_mutated_genes): 对选中的基因组，变异，有mutate_rate和mutation_intensity_coefficient（变异强度）
        def crossover(self, chosen_genes_1, chosen_genes_2)：两个基因之间交叉遗传，单点遗传
        def elitism_selection(self, number_of_elites): 根据fitness排列，最后下一代只有前number_of_elites个
        def roulette_wheel_selection(self, number_of_parents): 通过轮盘赌博，根据fittness的大小，选择下一代的父母"""

    def mutate(self, chosen_mutated_genes):
        # 设置一个变异率，具体方法是生成一个随机数组，其shape与chosen_mutated_genes一样，但都是由0-1之间
        # 于是就可以通过设定的变异率，小于这个设定的变异率的值，即可发生变异
        # 它会返回一个True、False组成的mutation_array，确定哪些基因会被突变
        mutation_array = np.random.random(chosen_mutated_genes.shape) < self.mutate_rate

        # 创建一个高斯分布并且size与被选中的要发生变异的基因组一样大，赋值给mutation
        mutation = np.random.normal(size=chosen_mutated_genes.shape)

        # 这里就是由突变强度系数控制的，生成一个突变值
        mutation_intensity_coefficient = 0.2
        scaled_mutation = mutation * mutation_intensity_coefficient

        # 最后再把scaled_mutation加到之前mutation_array确认出来为True的地方
        chosen_mutated_genes[mutation_array] += scaled_mutation[mutation_array]

    # 单点交叉：chosen_genes_1, chosen_genes_2是被选用于单点交叉的两个。
    # crossover_point 是随机选择的交叉点，在这个点之前的基因保持不变，在这个点及之后的基因将被交换。
    def crossover(self, chosen_genes_1, chosen_genes_2):
        crossover_point = np.random.randint(0, self.genes_len)
        chosen_genes_1[:crossover_point + 1], chosen_genes_2[:crossover_point + 1] = chosen_genes_2[
                                                                                     :crossover_point + 1], chosen_genes_1[
                                                                                                            :crossover_point + 1]

    def elitism_ranking_selection(self, number_of_elites):
        population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        return population[:number_of_elites]

    # 最经典的轮盘选择，即fitness越大，越容易被选
    # 实现：这里是通过累加的方式，即sum_of_all_fitness，然后就随机均匀分布出一个threshold，再遍历population，一旦大于，则选择其为父母
    def roulette_wheel_selection(self, number_of_parents):
        parents_list = []
        sum_of_all_fitness = sum(individual.fitness for individual in self.population)
        for _ in range(number_of_parents):
            threshold = np.random.uniform(0, sum_of_all_fitness)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > threshold:
                    parents_list.append(individual)
                    break

        return parents_list

    """最最最核心的操作：基于之前定义的四个工具，进行基因算法，返回通过实例化Individual得到的下一代实例的list"""

    def evolve(self):
        sum_score = 0
        for individual in self.population:
            individual.get_fitness()
            sum_score += individual.score
        # 这里的出来是给我在控制台观察训练情况的均分
        self.average_score_of_all_agents = round(sum_score / len(self.population), 2)

        # elitism_ranking_selection会自动筛选前多少位精英，并按照fitness排列
        self.population = self.elitism_ranking_selection(self.total_number_of_parents)
        # 由此，population的第一位自动就是best_individual
        self.best_individual = self.population[0]
        # 之前不是排序了嘛，现在再打乱
        random.shuffle(self.population)

        # 生成next_generation
        next_generation = []
        # 当还没有完成指定数量的小孩时
        while len(next_generation) < self.children_scale:
            # 使用轮盘赌，选出两个parents
            parent1, parent2 = self.roulette_wheel_selection(2)
            # 先复制，以免影响父母
            children1_genes, children2_genes = parent1.genes.copy(), parent2.genes.copy()

            # 根据之前定义的进行基因操作
            self.crossover(children1_genes, children2_genes)
            self.mutate(children1_genes)
            self.mutate(children2_genes)

            # 输入基因，包装成类
            children1_class = IntelligentAgent(children1_genes)
            children2_class = IntelligentAgent(children2_genes)
            next_generation.extend([children1_class, children2_class])

        random.shuffle(next_generation)
        # 打乱后输出由下一代组成的list
        self.population.extend(next_generation)

    def save_best(self):
        # 之前evolve里会得到一个self.population[0]
        current_best_score = self.best_individual.score
        saved_genes = os.path.join("GA_data", "saved_genes", "best_record", "reached_" + str(current_best_score))
        # 没有的话，生成相关的目录
        os.makedirs(os.path.dirname(saved_genes), exist_ok=True)
        # 覆盖并写入
        with open(saved_genes, "w") as f:
            for best_gene in self.best_individual.genes:
                f.write(str(best_gene) + " ")

        # 保留seed，以便复现
        saved_seed = os.path.join("GA_data", "reply_seed", "reached_" + str(current_best_score))
        os.makedirs(os.path.dirname(saved_seed), exist_ok=True)
        with open(saved_seed, "w") as f:
            f.write(str(self.best_individual.seed))

    # 每过一定轮数，保存那一代前100个的基因,观察训练情况
    def record_generation(self):
        record_gene_number = 100
        for individual in self.population:
            individual.get_fitness()
        chosen_population = self.elitism_ranking_selection(record_gene_number)
        for i in range(len(chosen_population)):
            pth = os.path.join("GA_data", "saved_genes", "first_100_genes", str(i))
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            with open(pth, "w") as f:
                for gene in chosen_population[i].genes:
                    f.write(str(gene) + " ")

