import time
from collections import deque
from typing import List, Tuple, Set, FrozenSet

def save_unobservable_solution(level_idx, path,elapsed_time):
    """Lưu lời giải tìm được vào file solutions.txt."""
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 4 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước Unobservable: {len(path)}\n")
        f.write(f"Path Unobservable: {path}\n\n")
    print(f"Unobservable: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_unobservable_search(level_data: List[List[str]], level_idx: int,
                                  possible_start_states: List[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]] = None):
    """
    Solve an unobservable Sokoban variant.

    If `possible_start_states` is None, the function will construct a single
    start state from `level_data` by finding the player position and the set
    of box positions and use that as the initial belief state. This makes the
    function backward-compatible with callers that pass only (level_data, level_idx).
    """

    def get_goals(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char in ['.', '+', '*']}

    def get_walls(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, char in enumerate(row) if char == '#'}

    goals = get_goals(level_data)
    walls = get_walls(level_data)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # --- Trạng thái niềm tin ban đầu ---
    if possible_start_states is None:
        # build a default single possible start state from the grid
        player = None
        boxes = []
        for y, row in enumerate(level_data):
            for x, ch in enumerate(row):
                if ch in ['@', '+']:
                    player = (x, y)
                if ch in ['$', '*']:
                    boxes.append((x, y))
        if player is None:
            # nothing we can do
            print(f"Unobservable: no player found in level {level_idx}")
            return None
        possible_start_states = [(player, frozenset(boxes))]

    initial_belief_state = frozenset(possible_start_states)

    # Hàng đợi cho BFS: (chuỗi_hành_động, trạng_thái_niềm_tin)
    queue = deque([([], initial_belief_state)])
    visited = {initial_belief_state}

    print(f"Unobservable: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    while queue:
        path, current_belief_state = queue.popleft()

        # --- Kiểm tra điều kiện thắng ---
        # Phải thắng trong TẤT CẢ các trạng thái trong tập hợp niềm tin
        if all(goals.issubset(s[1]) for s in current_belief_state):
            elapsed_time = time.time() - start_time
            print(f"Unobservable: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
            save_unobservable_solution(level_idx, path,elapsed_time)
            return path

        # --- Sinh các trạng thái niềm tin kế tiếp ---
        for action, (dx, dy) in enumerate(directions):
            next_belief_state_builder = set()

            # Áp dụng hành động cho MỌI trạng thái trong tập hợp niềm tin hiện tại
            for player_pos, boxes_pos in current_belief_state:
                next_player_pos = (player_pos[0] + dx, player_pos[1] + dy)
                if next_player_pos in walls:
                    next_belief_state_builder.add((player_pos, boxes_pos)) # Trạng thái không đổi
                    continue

                new_boxes_pos = boxes_pos
                if next_player_pos in boxes_pos:
                    next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                    if next_box_pos in walls or next_box_pos in boxes_pos:
                        next_belief_state_builder.add((player_pos, boxes_pos)) # Trạng thái không đổi
                        continue
                    box_list = list(boxes_pos)
                    box_list.remove(next_player_pos)
                    box_list.append(next_box_pos)
                    new_boxes_pos = frozenset(sorted(box_list))

                next_belief_state_builder.add((next_player_pos, new_boxes_pos))

            next_belief_state = frozenset(next_belief_state_builder)
            if next_belief_state not in visited:
                visited.add(next_belief_state)
                queue.append((path + [action], next_belief_state))

    print(f"Unobservable: Không tìm thấy lời giải cho Level {level_idx}.")
    return None
