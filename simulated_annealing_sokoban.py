import time
import math
import random
from typing import List, Tuple, Optional, Set, FrozenSet, Iterable
from collections import deque

def save_sa_solution(level_idx, path, elapsed_time):
    if path is None:
        return
    try:
        with open("solutions.txt", "a", encoding="utf-8") as f:
            f.write(f"--- Level {level_idx} ---\n")
            f.write("--- Nhóm 3 (Tối ưu) ---\n")
            f.write(f"Thời gian: {elapsed_time:.10f} giây\n")
            f.write(f"Số bước SA: {len(path)}\n")
            f.write(f"Path SA: {path}\n\n")
    except Exception:
        pass
    print(f"SA: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")


def _find_player(level_data: List[List[str]]) -> Optional[Tuple[int, int]]:
    for y, row in enumerate(level_data):
        for x, c in enumerate(row):
            if c in ['@', '+']:
                return (x, y)
    return None

def _get_goals(level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    return {(x, y) for y, row in enumerate(level_data) for x, c in enumerate(row) if c in ['.', '+', '*']}

def _get_boxes(level_data: List[List[str]]) -> FrozenSet[Tuple[int, int]]:
    return frozenset((x, y) for y, row in enumerate(level_data) for x, c in enumerate(row) if c in ['$', '*'])

def _get_walls(level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    return {(x, y) for y, row in enumerate(level_data) for x, c in enumerate(row) if c == '#'}

def _precompute_deadlocks(walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]], level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    deadlocks = set()
    height = len(level_data)
    width = len(level_data[0]) if height > 0 else 0

    for r in range(height):
        for c in range(width):
            pos = (c, r)
            if pos in walls or pos in goals:
                continue

            is_corner = ((c - 1, r) in walls and (c, r - 1) in walls) or \
                        ((c + 1, r) in walls and (c, r - 1) in walls) or \
                        ((c - 1, r) in walls and (c, r + 1) in walls) or \
                        ((c + 1, r) in walls and (c, r + 1) in walls)
            if is_corner:
                deadlocks.add(pos)
                continue

            for dy in [-1, 1]:
                if (c, r + dy) in walls:
                    is_stuck = True
                    for x_scan in range(c, -1, -1):
                        if (x_scan, r + dy) not in walls: is_stuck = False; break
                        if (x_scan, r) in goals: is_stuck = False; break
                    if not is_stuck: continue

                    is_stuck = True
                    for x_scan in range(c, width):
                        if (x_scan, r + dy) not in walls: is_stuck = False; break
                        if (x_scan, r) in goals: is_stuck = False; break
                    if is_stuck:
                        deadlocks.add(pos)

            for dx in [-1, 1]:
                if (c + dx, r) in walls:
                    is_stuck = True
                    for y_scan in range(r, -1, -1):
                        if (c + dx, y_scan) not in walls: is_stuck = False; break
                        if (c, y_scan) in goals: is_stuck = False; break
                    if not is_stuck: continue

                    is_stuck = True
                    for y_scan in range(r, height):
                        if (c + dx, y_scan) not in walls: is_stuck = False; break
                        if (c, y_scan) in goals: is_stuck = False; break
                    if is_stuck:
                        deadlocks.add(pos)

    return deadlocks

def energy_function(boxes: FrozenSet[Tuple[int, int]], goals: Set[Tuple[int, int]], deadlocks: Set[Tuple[int, int]]) -> int:
    if not goals:
        return 0

    for b in boxes:
        if b not in goals and b in deadlocks:
            return 100000

    total_manhattan_distance = 0
    goal_list = list(goals)
    for b in boxes:
        if b in goals:
            continue
        total_manhattan_distance += min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in goal_list)

    return total_manhattan_distance


def solve_with_simulated_annealing(level_data: List[List[str]], level_idx: int,
                                   possible_start_states: Optional[Iterable[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]]] = None,
                                   true_initial_state: Optional[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]] = None,
                                   initial_temp: Optional[float] = None,
                                   cooling_rate: float = 0.995,
                                   max_iterations: int = 50000,
                                   max_time: float = 10.0,
                                   restarts: int = 3) -> Optional[List[int]]:

    walls = _get_walls(level_data)
    goals = _get_goals(level_data)
    deadlocks = _precompute_deadlocks(walls, goals, level_data)

    if possible_start_states is None:
        player = _find_player(level_data)
        if player is None:
            print(f"SA: không tìm thấy người chơi cho Level {level_idx}")
            return None
        boxes = _get_boxes(level_data)
        possible_start_states = [(player, boxes)]

    possible_start_states = list(possible_start_states)
    if not possible_start_states:
        print(f"SA: không có trạng thái bắt đầu cho Level {level_idx}")
        return None

    start_state = true_initial_state if true_initial_state is not None else possible_start_states[0]

    action_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def player_bfs(start, goal, boxes_set: FrozenSet[Tuple[int, int]]):
        if start == goal:
            return [start]
        q = deque([(start, [start])])
        seen = {start}
        while q:
            pos, path = q.popleft()
            for dx, dy in directions:
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt in seen or nxt in walls or nxt in boxes_set:
                    continue
                if nxt == goal:
                    return path + [nxt]
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
        return None

    def coords_to_actions(path_coords: List[Tuple[int, int]]):
        out: List[int] = []
        for i in range(len(path_coords) - 1):
            dx = path_coords[i + 1][0] - path_coords[i][0]
            dy = path_coords[i + 1][1] - path_coords[i][1]
            out.append(action_map[(dx, dy)])
        return out

    calculate_energy = lambda boxes_pos: energy_function(boxes_pos, goals, deadlocks)

    if initial_temp is None:
        initial_temp = max(1.0, calculate_energy(start_state[1]) * 2.0)

    best_state = start_state
    best_energy = calculate_energy(best_state[1])
    best_actions: List[int] = []

    global_start = time.time()

    for attempt in range(max(1, restarts)):
        temp = initial_temp
        current_state = best_state
        current_energy = best_energy
        current_actions = best_actions.copy() if current_energy == best_energy else []

        for it in range(max_iterations):
            if time.time() - global_start > max_time:
                if best_energy == 0:
                    elapsed = time.time() - global_start
                    save_sa_solution(level_idx, best_actions, elapsed)
                    return best_actions
                print(f"SA: Đã hết thời gian cho Level {level_idx}")
                return None

            if current_energy == 0:
                elapsed = time.time() - global_start
                save_sa_solution(level_idx, current_actions, elapsed)
                return current_actions

            player_pos, boxes_pos = current_state
            candidates = []

            for b in boxes_pos:
                for dx, dy in directions:
                    push_from = (b[0] - dx, b[1] - dy)
                    dest = (b[0] + dx, b[1] + dy)
                    if dest in walls or dest in boxes_pos:
                        continue

                    path_to_push = player_bfs(player_pos, push_from, boxes_pos)
                    if path_to_push is None:
                        continue

                    new_boxes = set(boxes_pos)
                    new_boxes.remove(b)
                    new_boxes.add(dest)

                    next_state = (b, frozenset(new_boxes))
                    actions_seq = coords_to_actions(path_to_push) + [action_map[(dx, dy)]]
                    ne = calculate_energy(next_state[1])
                    candidates.append((actions_seq, next_state, ne))

            if not candidates:
                break

            candidates.sort(key=lambda t: t[2])
            topk = candidates[:min(10, len(candidates))]
            actions_seq, next_state, next_energy = random.choice(topk)

            delta = next_energy - current_energy
            if delta < 0 or (temp > 1e-9 and random.random() < math.exp(-delta / temp)):
                current_state = next_state
                current_energy = next_energy
                current_actions.extend(actions_seq)

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state
                    best_actions = current_actions.copy()

            temp *= cooling_rate

    if best_energy == 0:
        elapsed = time.time() - global_start
        save_sa_solution(level_idx, best_actions, elapsed)
        return best_actions

    print(f"SA: Không tìm thấy lời giải cho Level {level_idx}. Năng lượng tốt nhất={best_energy}")
    return None

