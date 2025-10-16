import time
import sys
from typing import List, Tuple, Optional, Set, FrozenSet

# Tăng giới hạn đệ quy
sys.setrecursionlimit(10000)

def save_ac_solution(level_idx, path, elapsed_time):
    """Lưu lời giải tìm được vào file solutions.txt."""
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 5 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước Arc Consistency: {len(path)}\n")
        f.write(f"Path Arc Consistency: {path}\n\n")
    print(f"Arc Consistency: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def is_deadlock(boxes_pos: FrozenSet[Tuple[int, int]], walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]]) -> bool:

    for box in boxes_pos:
        if box in goals:
            continue

        x, y = box

        # 1. Kiểm tra bế tắc ở góc (giống Forward Checking)
        is_stuck_in_corner = (
            ((x - 1, y) in walls and (x, y - 1) in walls) or
            ((x + 1, y) in walls and (x, y - 1) in walls) or
            ((x - 1, y) in walls and (x, y + 1) in walls) or
            ((x + 1, y) in walls and (x, y + 1) in walls)
        )
        if is_stuck_in_corner:
            return True

        # 2. Kiểm tra bế tắc "đóng băng" trên tường (tinh thần của Arc Consistency)
        # Kiểm tra tường ngang
        if (x, y - 1) in walls or (x, y + 1) in walls:
            is_frozen = True
            # Quét sang trái xem có đích không
            for i in range(x, -1, -1):
                if (i, y) in walls: break
                if (i, y) in goals: is_frozen = False; break
            if not is_frozen: continue

            is_frozen = True
             # Quét sang phải xem có đích không
            for i in range(x, len(walls)):
                if (i, y) in walls: break
                if (i, y) in goals: is_frozen = False; break
            if is_frozen: return True

        # Kiểm tra tường dọc
        if (x - 1, y) in walls or (x + 1, y) in walls:
            is_frozen = True
            # Quét lên trên
            for i in range(y, -1, -1):
                if (x, i) in walls: break
                if (x, i) in goals: is_frozen = False; break
            if not is_frozen: continue

            is_frozen = True
            # Quét xuống dưới
            for i in range(y, len(walls)):
                if (x, i) in walls: break
                if (x, i) in goals: is_frozen = False; break
            if is_frozen: return True

    return False

def solve_with_arc_consistency(level_data: List[List[str]], level_idx: int, max_depth=250):

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

    print(f"Arc Consistency: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _backtrack_with_ac(current_player_pos, current_boxes_pos, current_depth):
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
            if next_player_pos in walls: continue

            new_boxes_pos = current_boxes_pos
            is_push = False
            if next_player_pos in current_boxes_pos:
                is_push = True
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                if next_box_pos in walls or next_box_pos in current_boxes_pos: continue

                box_list = list(current_boxes_pos)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                new_boxes_pos = frozenset(sorted(box_list))
            if is_push and is_deadlock(new_boxes_pos, walls, goals):
                continue

            if _backtrack_with_ac(next_player_pos, new_boxes_pos, current_depth + 1):
                solution_path.insert(0, action)
                return True

        return False

    if _backtrack_with_ac(initial_player_pos, initial_boxes, 0):
        elapsed_time = time.time() - start_time
        print(f"Arc Consistency: Tìm thấy lời giải sau {elapsed_time:.10f} giây.")
        save_ac_solution(level_idx, solution_path,elapsed_time)
        return solution_path
    else:
        print(f"Arc Consistency: Không tìm thấy lời giải cho Level {level_idx}.")
        return None
