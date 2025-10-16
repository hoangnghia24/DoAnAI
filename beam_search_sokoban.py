import time
import heapq
from typing import List, Tuple, Optional, Set, FrozenSet


def save_beam_search_solution(level_idx, path,elapsed_time):
    """Lưu lời giải tìm được vào file beam_search_solutions.txt."""
    if path is None:
        return
    # Ghi nối đuôi vào file, mỗi lời giải cho một level
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write("--- Nhóm 3 ---\n")
        f.write(f"Thời gian: {elapsed_time:.10f} giây\n")
        f.write(f"Số bước Beam: {len(path)}\n")
        f.write(f"Path Beam: {path}\n\n")
    print(f"Beam Search: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def heuristic_manhattan_distance(boxes: FrozenSet[Tuple[int, int]], goals: Set[Tuple[int, int]]) -> int:
    """
    Heuristic tính tổng khoảng cách Manhattan nhỏ nhất từ mỗi thùng đến một đích.
    """
    total_distance = 0
    if not goals:
        return 0

    goal_list = list(goals)
    for box in boxes:
        min_dist_for_box = min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goal_list)
        total_distance += min_dist_for_box

    return total_distance

def solve_with_beam_search(level_data: List[List[str]], level_idx: int, beam_width=100, max_iterations=500):
    """
    Giải một màn chơi bằng Beam Search, trả về đường đi và lưu kết quả.
    """
    grid = [list(row) for row in level_data]

    # --- Các hàm helper nội bộ ---
    def find_player(g: List[List[str]]) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(g):
            for x, char in enumerate(row):
                if char in ['@', '+']:
                    return (x, y)
        return None

    def get_goals(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char in ['.', '+', '*']}

    def get_boxes(g: List[List[str]]) -> FrozenSet[Tuple[int, int]]:
        boxes = [(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char in ['$', '*']]
        return frozenset(sorted(boxes))

    def get_walls(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char == '#'}

    # --- Khởi tạo trạng thái ban đầu ---
    player_pos = find_player(grid)
    if player_pos is None:
        print(f"Beam Search: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(grid)
    goals = get_goals(grid)
    walls = get_walls(grid)
    initial_state = (player_pos, initial_boxes)

    # --- Cấu trúc dữ liệu cho thuật toán ---
    # beam chứa các tuple (path, state)
    beam = [([], initial_state)]
    visited = {initial_state}

    print(f"Beam Search: Bắt đầu giải Level {level_idx} (Beam Width = {beam_width})...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0: Trái, 1: Phải, 2: Lên, 3: Xuống

    # --- Vòng lặp chính của thuật toán ---
    for iteration in range(max_iterations):
        successors = []

        # 1. Mở rộng tất cả các trạng thái trong beam hiện tại
        for path, current_state in beam:
            current_player_pos, current_boxes = current_state

            # --- Kiểm tra điều kiện thắng ---
            if goals.issubset(current_boxes):
                elapsed_time = time.time() - start_time
                print(f"Beam Search: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
                print(f"  - Số bước đi: {len(path)}")
                print(f"  - Số trạng thái đã duyệt: {len(visited)}")
                save_beam_search_solution(level_idx, path,elapsed_time)
                return path

            # --- Sinh các trạng thái kế tiếp ---
            for action, (dx, dy) in enumerate(directions):
                next_player_pos = (current_player_pos[0] + dx, current_player_pos[1] + dy)

                if next_player_pos in walls:
                    continue

                new_boxes = current_boxes
                if next_player_pos in current_boxes:
                    next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                    if next_box_pos in walls or next_box_pos in current_boxes:
                        continue

                    box_list = list(current_boxes)
                    box_list.remove(next_player_pos)
                    box_list.append(next_box_pos)
                    new_boxes = frozenset(sorted(box_list))

                new_state = (next_player_pos, new_boxes)

                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [action]
                    successors.append((new_path, new_state))

        if not successors:
            print(f"Beam Search: Không còn trạng thái kế tiếp để mở rộng ở vòng lặp {iteration + 1}.")
            break

        # 2. Chọn ra `beam_width` trạng thái tốt nhất từ tất cả các trạng thái con
        beam = heapq.nsmallest(
            beam_width,
            successors,
            key=lambda item: heuristic_manhattan_distance(item[1][1], goals) # item[1][1] là boxes_pos
        )

    print(f"Beam Search: Không tìm thấy lời giải cho Level {level_idx} trong giới hạn {max_iterations} vòng lặp.")
    return None
