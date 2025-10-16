import time
import heapq
from typing import List, Tuple, Optional, Set, FrozenSet


def save_greedy_solution(level_idx, path,elapsed_time):
    if path is None:
        return
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

    goal_list = list(goals)

    for box in boxes:
        min_dist_for_box = min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goal_list)
        total_distance += min_dist_for_box

    return total_distance

def solve_with_greedy(level_data: List[List[str]], level_idx: int, max_states=50000):
    grid = [list(row) for row in level_data]

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

    player_pos = find_player(grid)
    if player_pos is None:
        print(f"Greedy: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(grid)
    goals = get_goals(grid)
    walls = get_walls(grid)
    initial_state = (player_pos, initial_boxes)

    pq = [(heuristic_manhattan_distance(initial_boxes, goals), [], initial_state)]
    heapq.heapify(pq)
    visited = {initial_state}

    print(f"Greedy: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        if len(visited) > max_states:
            print(f"Greedy: Đã vượt quá {max_states} trạng thái. Dừng lại.")
            return None

        _, path, current_state = heapq.heappop(pq)
        current_player_pos, current_boxes = current_state

        if goals.issubset(current_boxes):
            elapsed_time = time.time() - start_time
            print(f"Greedy: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
            print(f"  - Số bước đi: {len(path)}")
            print(f"  - Số trạng thái đã duyệt: {len(visited)}")
            save_greedy_solution(level_idx, path,elapsed_time)
            return path

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
                heuristic_value = heuristic_manhattan_distance(new_boxes, goals)
                heapq.heappush(pq, (heuristic_value, new_path, new_state))

    print(f"Greedy: Không tìm thấy lời giải cho Level {level_idx}.")
    return None

