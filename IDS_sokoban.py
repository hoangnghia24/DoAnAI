import time
import sys
from typing import List, Tuple, Optional, Set, FrozenSet

# Tăng giới hạn đệ quy để xử lý các màn chơi phức tạp
sys.setrecursionlimit(10000)

def save_ids_solution(level_idx, path, elapsed_time):
    """Lưu lời giải tìm được vào file ids_solutions.txt."""
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 1 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước: {len(path)}\n")
        f.write(f"Path IDS: {path}\n\n")
    print(f"IDS: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_ids(level_data: List[List[str]], level_idx: int, max_depth=150):
    """
    Giải một màn chơi Sokoban bằng thuật toán Iterative Deepening Search (IDS).
    """
    # --- Các hàm helper nội bộ và khởi tạo ---
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
    if initial_player_pos is None:
        print(f"IDS: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(level_data)
    goals = get_goals(level_data)
    walls = get_walls(level_data)

    print(f"IDS: Bắt đầu giải Level {level_idx} (max_depth={max_depth})...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0: Trái, 1: Phải, 2: Lên, 3: Xuống

    # --- Hàm đệ quy cho DLS ---
    def _dls_recursive(path, current_player_pos, current_boxes_pos, depth_limit):
        # Điều kiện dừng: Vượt quá giới hạn độ sâu
        if len(path) >= depth_limit:
            return None

        # Thử tất cả các hành động có thể
        for action, (dx, dy) in enumerate(directions):
            next_player_pos = (current_player_pos[0] + dx, current_player_pos[1] + dy)

            if next_player_pos in walls:
                continue

            new_boxes_pos = current_boxes_pos
            if next_player_pos in current_boxes_pos:
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                if next_box_pos in walls or next_box_pos in current_boxes_pos:
                    continue

                box_list = list(current_boxes_pos)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                new_boxes_pos = frozenset(sorted(box_list))

            new_path = path + [action]
            new_state = (next_player_pos, new_boxes_pos)

            # Tránh lặp lại trạng thái trong cùng một đường đi
            if new_state in visited_this_iteration:
                continue

            visited_this_iteration.add(new_state)

            # Kiểm tra điều kiện thắng
            if goals.issubset(new_boxes_pos):
                return new_path

            # Gọi đệ quy
            result = _dls_recursive(new_path, next_player_pos, new_boxes_pos, depth_limit)
            if result is not None:
                return result

        return None

    # --- Vòng lặp chính của IDS ---
    for depth_limit in range(max_depth):

        # visited phải được reset ở mỗi vòng lặp của IDS
        visited_this_iteration = {(initial_player_pos, initial_boxes)}

        solution_path = _dls_recursive([], initial_player_pos, initial_boxes, depth_limit)

        if solution_path is not None:
            elapsed_time = time.time() - start_time
            print(f"IDS: Tìm thấy lời giải tối ưu sau {elapsed_time:.2f} giây.")
            print(f"  - Số bước đi: {len(solution_path)}")
            save_ids_solution(level_idx, solution_path,elapsed_time)
            return solution_path

    print(f"IDS: Không tìm thấy lời giải cho Level {level_idx} trong giới hạn độ sâu.")
    return None
