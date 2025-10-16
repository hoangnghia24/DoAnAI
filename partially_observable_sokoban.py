import time
import heapq  # Sử dụng heapq cho hàng đợi ưu tiên của A*
from collections import deque
from typing import List, Tuple, Optional, Set, FrozenSet

# --- Các hàm phụ trợ (Lưu lời giải, lấy tường, đích, v.v.) giữ nguyên ---
def save_partially_observable_solution(level_idx: int, path: Optional[List[int]], elapsed_time: float):
    if path is None:
        return
    try:
        with open("solutions.txt", "a", encoding="utf-8") as f:
            f.write(f"--- Level {level_idx} ---\n")
            f.write("--- Nhóm 4 (Tối ưu A*) ---\n")
            f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
            f.write(f"Số bước Partially Observable: {len(path)}\n")
            f.write(f"Path Partially Observable: {path}\n\n")
        print(f"Partially Observable A*: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")
    except Exception:
        pass

def get_goals(g: List[List[str]]) -> Set[Tuple[int, int]]:
    return {(x, y) for y, row in enumerate(g) for x, c in enumerate(row) if c in ['.', '+', '*']}

def get_walls(g: List[List[str]]) -> Set[Tuple[int, int]]:
    return {(x, y) for y, row in enumerate(g) for x, c in enumerate(row) if c == '#'}

# Tái sử dụng hàm tính deadlock hiệu quả
def _precompute_deadlocks(walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]], level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    deadlocks = set()
    height = len(level_data)
    width = len(level_data[0]) if height > 0 else 0
    for r in range(height):
        for c in range(width):
            pos = (c, r)
            if pos in walls or pos in goals:
                continue
            # Góc
            if ((c - 1, r) in walls and (c, r - 1) in walls) or \
               ((c + 1, r) in walls and (c, r - 1) in walls) or \
               ((c - 1, r) in walls and (c, r + 1) in walls) or \
               ((c + 1, r) in walls and (c, r + 1) in walls):
                deadlocks.add(pos)
    return deadlocks


# --- TỐI ƯU HÓA BẮT ĐẦU TỪ ĐÂY ---

def heuristic_for_belief_state(belief: FrozenSet[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]], 
                               goals: Set[Tuple[int, int]], 
                               deadlocks: Set[Tuple[int, int]]) -> int:
    """
    Hàm Heuristic cho thuật toán A*.
    Ước tính chi phí từ tập hợp niềm tin hiện tại đến mục tiêu.
    Chúng ta chọn cách tiếp cận lạc quan: lấy heuristic nhỏ nhất trong số tất cả các trạng thái có thể.
    """
    min_heuristic = float('inf')
    
    goal_list = list(goals)
    if not goal_list:
        return 0

    for _, boxes_pos in belief:
        current_h = 0
        
        # 1. Tổng khoảng cách Manhattan
        for box in boxes_pos:
            if box not in goals:
                current_h += min(abs(box[0] - g[0]) + abs(box[1] - g[1]) for g in goal_list)

        # 2. Phạt nặng cho deadlock
        for box in boxes_pos:
            if box not in goals and box in deadlocks:
                current_h += 1000  # Phạt nặng
                break # Một hộp bị kẹt là đủ tệ

        if current_h < min_heuristic:
            min_heuristic = current_h
            
    return min_heuristic if min_heuristic != float('inf') else 0

def solve_with_partially_observable_search_astar(level_data: List[List[str]], level_idx: int,
                                                 true_initial_state: Optional[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]] = None,
                                                 possible_start_states: Optional[List[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]]] = None,
                                                 max_steps: int = 20000,
                                                 max_time_s: float = 30.0) -> Optional[List[int]]:
    """
    Giải quyết Sokoban quan sát được một phần bằng tìm kiếm A* trên không gian tập hợp niềm tin.
    """
    
    def make_default_states():
        player = None
        boxes = []
        for y, row in enumerate(level_data):
            for x, c in enumerate(row):
                if c in ['@', '+']: player = (x, y)
                if c in ['$', '*']: boxes.append((x, y))
        if player is None: return None, None
        state = (player, frozenset(boxes))
        return state, [state]

    def observation(player_pos: Tuple[int, int], walls: Set[Tuple[int, int]], boxes: FrozenSet[Tuple[int, int]]):
        obs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = (player_pos[0] + dx, player_pos[1] + dy)
                if p == player_pos: obs.append('@')
                elif p in walls: obs.append('#')
                elif p in boxes: obs.append('$')
                else: obs.append(' ')
        return tuple(obs)

    goals = get_goals(level_data)
    walls = get_walls(level_data)
    deadlocks = _precompute_deadlocks(walls, goals, level_data)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # L, R, U, D

    if true_initial_state is None or possible_start_states is None:
        default_state, default_list = make_default_states()
        if default_state is None:
            print(f"Partially Observable A*: không tìm thấy người chơi ở level {level_idx}")
            return None
        if possible_start_states is None:
            possible_start_states = default_list

    start_belief = frozenset(possible_start_states)
    
    # Hàng đợi ưu tiên cho A*: (f_cost, g_cost, path, belief)
    # g_cost (len(path)) được thêm vào để phá vỡ thế hòa bằng cách ưu tiên các đường đi ngắn hơn
    h_cost = heuristic_for_belief_state(start_belief, goals, deadlocks)
    priority_queue = [(h_cost, 0, [], start_belief)] 
    
    # visited bây giờ lưu chi phí tốt nhất để đến một belief state
    visited = {start_belief: 0}

    print(f"Partially Observable A*: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    step = 0
    while priority_queue and step < max_steps:
        if time.time() - start_time > max_time_s:
            print(f"Partially Observable A*: Đã hết thời gian ({max_time_s}s).")
            return None

        _, g_cost, path, belief = heapq.heappop(priority_queue)
        step += 1

        if g_cost > visited[belief]:
            continue

        if all(goals.issubset(b[1]) for b in belief):
            elapsed = time.time() - start_time
            save_partially_observable_solution(level_idx, path, elapsed)
            return path

        for action, (dx, dy) in enumerate(directions):
            # Tạo tập hợp niềm tin kế thừa
            successor_builder = set()
            for player_pos, boxes_pos in belief:
                next_player = (player_pos[0] + dx, player_pos[1] + dy)
                
                if next_player in walls:
                    successor_builder.add((player_pos, boxes_pos))
                    continue
                
                if next_player in boxes_pos:
                    next_box = (next_player[0] + dx, next_player[1] + dy)
                    if next_box in walls or next_box in boxes_pos:
                        successor_builder.add((player_pos, boxes_pos)) # Đẩy không thành công
                    else:
                        new_boxes = set(boxes_pos)
                        new_boxes.remove(next_player)
                        new_boxes.add(next_box)
                        successor_builder.add((next_player, frozenset(new_boxes)))
                else:
                    successor_builder.add((next_player, boxes_pos))
            
            # Phân nhóm các trạng thái kế thừa theo quan sát
            obs_groups = {}
            for st in successor_builder:
                obs = observation(st[0], walls, st[1])
                obs_groups.setdefault(obs, set()).add(st)

            # Đưa các tập hợp niềm tin mới vào hàng đợi
            for _, group in obs_groups.items():
                new_belief = frozenset(group)
                new_g_cost = g_cost + 1

                if new_belief not in visited or new_g_cost < visited[new_belief]:
                    visited[new_belief] = new_g_cost
                    h_cost = heuristic_for_belief_state(new_belief, goals, deadlocks)
                    f_cost = new_g_cost + h_cost
                    new_path = path + [action]
                    heapq.heappush(priority_queue, (f_cost, new_g_cost, new_path, new_belief))

    print(f"Partially Observable A*: Không tìm thấy lời giải trong giới hạn ({max_steps} bước).")
    return None