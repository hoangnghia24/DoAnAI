import time
import sys
from typing import List, Tuple, Optional, Set, FrozenSet

sys.setrecursionlimit(10000)

def save_fc_solution(level_idx, path,elapsed_time):
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 5 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước Forward Checking: {len(path)}\n")
        f.write(f"Path Forward Checking: {path}\n\n")
    print(f"Forward Checking: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def is_deadlock(boxes_pos: FrozenSet[Tuple[int, int]], walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]]) -> bool:
    for box in boxes_pos:
        if box not in goals:
            x, y = box
            is_stuck_in_corner = (
                (x - 1, y) in walls and (x, y - 1) in walls or
                (x + 1, y) in walls and (x, y - 1) in walls or
                (x - 1, y) in walls and (x, y + 1) in walls or
                (x + 1, y) in walls and (x, y + 1) in walls
            )
            if is_stuck_in_corner:
                return True
    return False

def solve_with_forward_checking(level_data: List[List[str]], level_idx: int, max_depth=250):
    def find_player(g: List[List[str]]) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(g):
            for x, char in enumerate(row):
                if char in ['@', '+']: return (x, y)
        return None

    def get_goals(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char in ['.', '+', '*']}

    def get_boxes(g: List[List[str]]) -> FrozenSet[Tuple[int, int]]:
        return frozenset(sorted([(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char in ['$', '*']]))

    def get_walls(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char == '#'}

    initial_player_pos = find_player(level_data)
    initial_boxes = get_boxes(level_data)
    goals = get_goals(level_data)
    walls = get_walls(level_data)
    visited = set()
    solution_path = []

    print(f"Forward Checking: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _backtrack_with_fc(current_player_pos, current_boxes_pos, current_depth):
        if goals.issubset(current_boxes_pos):
            return True
        if current_depth >= max_depth:
            return False

        current_state = (current_player_pos, current_boxes_pos)
        if current_state in visited:
            return False
        visited.add(current_state)

        for action, (dx, dy) in enumerate(directions):
            next_player_pos = (current_player_pos[0] + dx, current_player_pos[1] + dy)

            if next_player_pos in walls:
                continue

            new_boxes_pos = current_boxes_pos
            is_push = False

            if next_player_pos in current_boxes_pos:
                is_push = True
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                if next_box_pos in walls or next_box_pos in current_boxes_pos:
                    continue

                box_list = list(current_boxes_pos)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                new_boxes_pos = frozenset(sorted(box_list))

            if is_push and is_deadlock(new_boxes_pos, walls, goals):
                continue

            if _backtrack_with_fc(next_player_pos, new_boxes_pos, current_depth + 1):
                solution_path.insert(0, action)
                return True

        return False

    if _backtrack_with_fc(initial_player_pos, initial_boxes, 0):
        elapsed_time = time.time() - start_time
        print(f"Forward Checking: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
        print(f"  - Số bước đi: {len(solution_path)}")
        save_fc_solution(level_idx, solution_path,elapsed_time)
        return solution_path
    else:
        print(f"Forward Checking: Không tìm thấy lời giải cho Level {level_idx}.")
        return None

