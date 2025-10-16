import time
import sys
from typing import List, Tuple, Optional, Set, FrozenSet

# Tăng giới hạn đệ quy để xử lý các màn chơi phức tạp
sys.setrecursionlimit(10000)


def save_dls_solution(level_idx, path,elapsed_time):
    """Lưu lời giải tìm được vào file solutions.txt."""
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 1 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước: {len(path)}\n")
        f.write(f"Path DLS: {path}\n\n")
    print(f"DLS: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_dls(level_data: List[List[str]], level_idx: int, depth_limit=30):
    """
    Giải một màn chơi Sokoban bằng thuật toán Depth-Limited Search (DLS).
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
        print(f"DLS: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(level_data)
    goals = get_goals(level_data)
    walls = get_walls(level_data)

    # Visited dùng để tránh các vòng lặp vô hạn
    visited = set()

    print(f"DLS: Bắt đầu giải Level {level_idx} (depth_limit={depth_limit})...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0: Trái, 1: Phải, 2: Lên, 3: Xuống

    # --- Hàm đệ quy cốt lõi ---
    def _dls_recursive(current_player_pos, current_boxes_pos, current_depth):
        # 1. Điều kiện dừng: Vượt quá giới hạn độ sâu
        if current_depth > depth_limit:
            return None # "cutoff"

        # 2. Điều kiện dừng: Trạng thái đã được duyệt qua (tránh lặp)
        current_state = (current_player_pos, current_boxes_pos)
        if current_state in visited:
            return None
        visited.add(current_state)

        # 3. Thử tất cả các hành động có thể
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

            # 4. Kiểm tra điều kiện thắng sau khi di chuyển
            if goals.issubset(new_boxes_pos):
                return [action] # Tìm thấy lời giải, trả về hành động cuối cùng

            # 5. Gọi đệ quy cho trạng thái mới
            result_path = _dls_recursive(next_player_pos, new_boxes_pos, current_depth + 1)

            # Nếu nhánh con tìm thấy lời giải, xây dựng đường đi và trả về
            if result_path is not None:
                return [action] + result_path

        # 6. Quay lui: Xóa trạng thái khỏi visited để các nhánh khác có thể đi qua nó
        # (Quan trọng trong DLS và các thuật toán tìm kiếm lặp)
        visited.remove(current_state)
        return None

    # --- Gọi hàm và xử lý kết quả ---
    solution_path = _dls_recursive(initial_player_pos, initial_boxes, 0)

    if solution_path is not None:
        elapsed_time = time.time() - start_time
        print(f"DLS: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
        print(f"  - Số bước đi: {len(solution_path)}")
        save_dls_solution(level_idx, solution_path,elapsed_time)
        return solution_path
    else:
        print(f"DLS: Không tìm thấy lời giải cho Level {level_idx} trong giới hạn độ sâu.")
        return None
