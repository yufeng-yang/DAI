class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # 距离起点的距离
        self.h = 0  # 启发式的估算到终点的距离
        self.f = 0  # 总的分数 G + H

def heuristic(a, b):
    # For grids, use Manhattan distance judgment
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 该方法在尝试获取输入的节点的四个相邻节点
def get_neighbors(node_position, grid):
    # node_position：输入的节点，也就是当前探索到的那个节点
    (x, y) = node_position
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    # print(neighbors)
    valid_neighbors = [n for n in neighbors if 0 <= n[0] < grid and 0 <= n[1] < grid]
    # print(valid_neighbors)
    return valid_neighbors

# a*算法的发挥，传入网格，开始位置，目标位置和蛇身
def a_star_algorithm(grid, start, end, snake_body):
    open_set = set()
    closed_set = set()
    start_node = Node(None, start)
    end_node = Node(None, end)

    # 初始化起始节点的 G, H 和 F 值
    start_node.g = 0
    start_node.h = heuristic(start, end)
    start_node.f = start_node.g + start_node.h

    # 将起始节点添加到开启列表
    open_set.add(start_node)

    while open_set:
        # 从开启列表中找到 F 值最小的节点
        current_node = min(open_set, key=lambda o: o.f)
        open_set.remove(current_node)
        closed_set.add(current_node)

        # 检查是否达到终点
        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # 返回路径

        # 获取当前节点的所有相邻节点
        neighbors = get_neighbors(current_node.position, grid)
        for next in neighbors:
            if next in snake_body or any(node.position == next for node in closed_set):
                continue  # 如果是蛇身或已经处理过的节点，则跳过

            # 创建相邻节点
            neighbor = Node(current_node, next)
            if neighbor in open_set:
                continue  # 如果已在开启列表，跳过

            # 计算相邻节点的 G, H 和 F 值
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(next, end)
            neighbor.f = neighbor.g + neighbor.h

            # 将相邻节点添加到开启列表
            open_set.add(neighbor)

    return None  # 如果没有路径
