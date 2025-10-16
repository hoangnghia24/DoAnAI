import time
import heapq
from typing import List, Tuple, Optional, Set, FrozenSet

COST_PLAYER_MOVE = 1
COST_BOX_PUSH = 10

def save_ucs_solution(level_idx, path,elapsed_time):
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write(f"--- Nhóm 2 ---\n")
        f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
        f.write(f"Số bước: {len(path)}\n")
        f.write(f"Path UCS: {path}\n\n")
    print(f"UCS: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_ucs(level_data: List[List[str]], level_idx: int, max_states=50000):
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
        print(f"UCS: Không tìm thấy người chơi ở Level {level_idx}")
        return None

    initial_boxes = get_boxes(level_data)
    goals = get_goals(level_data)
    walls = get_walls(level_data)
    initial_state = (initial_player_pos, initial_boxes)

    pq = [(0, [], initial_state)]
    heapq.heapify(pq)

    visited = {initial_state: 0}

    print(f"UCS: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        if len(visited) > max_states:
            print(f"UCS: Đã vượt quá {max_states} trạng thái. Dừng lại.")
            return None

        current_cost, path, current_state = heapq.heappop(pq)
        current_player_pos, current_boxes = current_state

        if current_cost > visited[current_state]:
            continue

        if goals.issubset(current_boxes):
            elapsed_time = time.time() - start_time
            print(f"UCS: Tìm thấy lời giải sau {elapsed_time:.10f} giây.")
            print(f"  - Tổng chi phí: {current_cost}")
            print(f"  - Số bước đi: {len(path)}")
            save_ucs_solution(level_idx, path,elapsed_time)
            return path

        for action, (dx, dy) in enumerate(directions):
            next_player_pos = (current_player_pos[0] + dx, current_player_pos[1] + dy)

            if next_player_pos in walls:
                continue

            new_boxes = current_boxes
            move_cost = COST_PLAYER_MOVE

            if next_player_pos in current_boxes:
                move_cost = COST_BOX_PUSH
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                if next_box_pos in walls or next_box_pos in current_boxes:
                    continue

                box_list = list(current_boxes)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                new_boxes = frozenset(sorted(box_list))

            new_cost = current_cost + move_cost
            new_state = (next_player_pos, new_boxes)

            if new_state not in visited or new_cost < visited[new_state]:
                visited[new_state] = new_cost
                new_path = path + [action]
                heapq.heappush(pq, (new_cost, new_path, new_state))

    print(f"UCS: Không tìm thấy lời giải cho Level {level_idx}.")
    return None

