# Tên file: bfs_sokoban.py

import time
from collections import deque
from typing import List, Tuple, Optional

# NOTE: Avoid importing Game or SokobanEnv here to prevent circular imports when
# this module is used from `sokoban.py` (which imports this module). Instead
# implement a small, self-contained grid simulator for BFS so the solver can
# be started safely in a background thread.

def save_bfs_solution(level_idx, path,elapsed_time):
    """Lưu lời giải tìm được vào file bfs_solutions.txt."""
    if path is None:
        return
    # Ghi nối đuôi vào file, mỗi lời giải cho một level
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 1 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước BFS: {len(path)}\n")
        f.write(f"Path BFS: {path}\n\n")
    print(f"BFS: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_bfs(level_data, level_idx, max_states=50000):
    """
    Giải một màn chơi bằng BFS, trả về đường đi và lưu kết quả vào file.
    """
    # Normalize level_data to list of lists
    grid = [list(row) for row in level_data]

    def find_player(g: List[List[str]]) -> Optional[Tuple[int,int]]:
        for y, row in enumerate(g):
            for x, c in enumerate(row):
                if c in ['@', '+']:
                    return (x, y)
        return None

    def get_goals(g: List[List[str]]):
        goals = set()
        for y, row in enumerate(g):
            for x, c in enumerate(row):
                if c in ['.', '+', '*']:
                    goals.add((x, y))
        return goals

    def get_boxes(g: List[List[str]]):
        boxes = []
        for y, row in enumerate(g):
            for x, c in enumerate(row):
                if c in ['$', '*']:
                    boxes.append((x, y))
        return tuple(sorted(boxes))

    player = find_player(grid)
    if player is None:
        print(f"BFS: No player found on level {level_idx}")
        return None

    goals = get_goals(grid)
    initial_state = {'player': player, 'boxes': get_boxes(grid)}

    # Hàng đợi (queue) cho BFS
    queue = deque([(initial_state, [])])
    visited = set([(tuple(initial_state['player']), initial_state['boxes'])])
    
    print(f"BFS: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    # Helpers to apply actions on a (player, boxes) state
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def cell_at(x: int, y: int) -> str:
        if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
            return '#'
        return grid[y][x]

    def apply_action(state, action: int):
        px, py = state['player']
        boxes = list(state['boxes'])
        dx, dy = directions[action]
        nx, ny = px + dx, py + dy
        # check wall
        if cell_at(nx, ny) == '#':
            return None
        # check if box present
        if (nx, ny) in boxes:
            bx, by = nx + dx, ny + dy
            if cell_at(bx, by) == '#':
                return None
            if (bx, by) in boxes:
                return None
            # push box
            boxes = [(bx, by) if b == (nx, ny) else b for b in boxes]
        # move player
        new_state = {'player': (nx, ny), 'boxes': tuple(sorted(boxes))}
        return new_state
    
    while queue:
        if len(visited) > max_states:
            print(f"BFS: Đã vượt quá {max_states} trạng thái. Dừng lại.")
            return None

        current_state, path = queue.popleft()

        # Kiểm tra điều kiện thắng
        if set(current_state['boxes']) <= goals and len(goals) > 0:
            elapsed_time = time.time() - start_time
            print(f"BFS: Tìm thấy lời giải ngắn nhất sau {elapsed_time:.10f} giây.")
            save_bfs_solution(level_idx, path,elapsed_time)
            return path
            
        # Thử 4 hướng di chuyển
        for action in range(4):
            new_state = apply_action(current_state, action)
            if new_state is None:
                continue
            key = (tuple(new_state['player']), new_state['boxes'])
            if key in visited:
                continue
            visited.add(key)
            queue.append((new_state, path + [action]))
    
    print("BFS: Không tìm thấy lời giải.")
    return None