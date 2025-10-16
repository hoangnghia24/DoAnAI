import time
import heapq
from typing import List, Tuple, Optional, Set, FrozenSet


def save_greedy_solution(level_idx, path,elapsed_time):
    """Lưu lời giải tìm được vào file greedy_solutions.txt."""
    if path is None:
        return
    # Ghi nối đuôi vào file, mỗi lời giải cho một level
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 2 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước: {len(path)}\n")
        f.write(f"Path Greedy: {path}\n\n")
    print(f"Greedy: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def heuristic_manhattan_distance(boxes: FrozenSet[Tuple[int, int]], goals: Set[Tuple[int, int]]) -> int:

    total_distance = 0
    if not goals:
        return 0

    # Chuyển set thành list để tối ưu việc lặp
    goal_list = list(goals)

    for box in boxes:
        # Tìm khoảng cách ngắn nhất từ thùng hiện tại đến bất kỳ đích nào
        min_dist_for_box = min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goal_list)
        total_distance += min_dist_for_box

    return total_distance

def solve_with_greedy(level_data: List[List[str]], level_idx: int, max_states=50000):
    """
    Giải một màn chơi bằng Greedy Best-First Search, trả về đường đi và lưu kết quả.
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
        print(f"Greedy: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(grid)
    goals = get_goals(grid)
    walls = get_walls(grid)
    initial_state = (player_pos, initial_boxes)

    # --- Cấu trúc dữ liệu cho thuật toán ---
    # Hàng đợi ưu tiên: (heuristic, path, state)
    pq = [(heuristic_manhattan_distance(initial_boxes, goals), [], initial_state)]
    heapq.heapify(pq)
    visited = {initial_state}

    print(f"Greedy: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0: Trái, 1: Phải, 2: Lên, 3: Xuống

    # --- Vòng lặp chính của thuật toán ---
    while pq:
        if len(visited) > max_states:
            print(f"Greedy: Đã vượt quá {max_states} trạng thái. Dừng lại.")
            return None

        _, path, current_state = heapq.heappop(pq)
        current_player_pos, current_boxes = current_state

        # --- Kiểm tra điều kiện thắng ---
        # Nếu tất cả các vị trí đích đều có thùng
        if goals.issubset(current_boxes):
            elapsed_time = time.time() - start_time
            print(f"Greedy: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
            print(f"  - Số bước đi: {len(path)}")
            print(f"  - Số trạng thái đã duyệt: {len(visited)}")
            save_greedy_solution(level_idx, path,elapsed_time)
            return path

        # --- Sinh các trạng thái kế tiếp ---
        for action, (dx, dy) in enumerate(directions):
            next_player_pos = (current_player_pos[0] + dx, current_player_pos[1] + dy)

            # 1. Người chơi di chuyển vào tường -> không hợp lệ
            if next_player_pos in walls:
                continue

            new_boxes = current_boxes

            # 2. Người chơi đẩy thùng
            if next_player_pos in current_boxes:
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)

                # Thùng bị đẩy vào tường hoặc một thùng khác -> không hợp lệ
                if next_box_pos in walls or next_box_pos in current_boxes:
                    continue

                # Cập nhật vị trí các thùng
                box_list = list(current_boxes)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                new_boxes = frozenset(sorted(box_list))

            # --- Tạo trạng thái mới và thêm vào hàng đợi ---
            new_state = (next_player_pos, new_boxes)

            if new_state not in visited:
                visited.add(new_state)
                new_path = path + [action]
                heuristic_value = heuristic_manhattan_distance(new_boxes, goals)
                heapq.heappush(pq, (heuristic_value, new_path, new_state))

    print(f"Greedy: Không tìm thấy lời giải cho Level {level_idx}.")
    return None
