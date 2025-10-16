import time
import sys
from typing import List, Tuple, Optional, Set, FrozenSet

# Tăng giới hạn đệ quy để xử lý các màn chơi phức tạp
sys.setrecursionlimit(10000)

def save_backtracking_solution(level_idx, path,elapsed_time):
    """Lưu lời giải tìm được vào file backtracking_solutions.txt."""
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 5 ---\n")
        f.write(f"Thời gian: {elapsed_time:.10f} giây\n")
        f.write(f"Số bước Backtracking: {len(path)}\n")
        f.write(f"Path Backtracking: {path}\n\n")
    print(f"Backtracking: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_backtracking(level_data: List[List[str]], level_idx: int, max_depth=250):
    """
    Giải một màn chơi Sokoban bằng thuật toán Backtracking (DFS).
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
        print(f"Backtracking: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(level_data)
    goals = get_goals(level_data)
    walls = get_walls(level_data)

    visited = set()
    solution_path = []

    print(f"Backtracking: Bắt đầu giải Level {level_idx} (max_depth={max_depth})...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0: Trái, 1: Phải, 2: Lên, 3: Xuống

    # --- Hàm đệ quy cốt lõi ---
    def _backtrack(current_player_pos, current_boxes_pos, current_depth):
        # 1. Điều kiện dừng: Thắng
        if goals.issubset(current_boxes_pos):
            return True

        # 2. Điều kiện dừng: Vượt quá độ sâu tìm kiếm
        if current_depth >= max_depth:
            return False

        # 3. Điều kiện dừng: Trạng thái đã được duyệt qua
        current_state = (current_player_pos, current_boxes_pos)
        if current_state in visited:
            return False
        visited.add(current_state)

        # 4. Thử tất cả các hành động có thể
        for action, (dx, dy) in enumerate(directions):
            next_player_pos = (current_player_pos[0] + dx, current_player_pos[1] + dy)

            # Người chơi đi vào tường -> bỏ qua
            if next_player_pos in walls:
                continue

            new_boxes_pos = current_boxes_pos
            # Người chơi đẩy thùng
            if next_player_pos in current_boxes_pos:
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)

                # Thùng bị đẩy vào tường hoặc thùng khác -> bỏ qua
                if next_box_pos in walls or next_box_pos in current_boxes_pos:
                    continue

                # Cập nhật vị trí các thùng
                box_list = list(current_boxes_pos)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                new_boxes_pos = frozenset(sorted(box_list))

            # 5. Gọi đệ quy cho trạng thái mới
            if _backtrack(next_player_pos, new_boxes_pos, current_depth + 1):
                # Nếu lời giải được tìm thấy ở nhánh con, thêm hành động vào lời giải
                solution_path.insert(0, action)
                return True

        # 6. Quay lui: Nếu không có hành động nào dẫn đến giải pháp
        return False

    # --- Gọi hàm và xử lý kết quả ---
    if _backtrack(initial_player_pos, initial_boxes, 0):
        elapsed_time = time.time() - start_time
        print(f"Backtracking: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
        print(f"  - Số bước đi: {len(solution_path)}")
        save_backtracking_solution(level_idx, solution_path,elapsed_time)
        return solution_path
    else:
        print(f"Backtracking: Không tìm thấy lời giải cho Level {level_idx} trong giới hạn độ sâu.")
        return None
