import time
import math
import random
from typing import List, Tuple, Optional, Set, FrozenSet, Iterable
from collections import deque

# --- Các hàm phụ trợ không thay đổi ---
def save_sa_solution(level_idx, path, elapsed_time):
    # ... (giữ nguyên mã của bạn)
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
    # ... (giữ nguyên mã của bạn)
    for y, row in enumerate(level_data):
        for x, c in enumerate(row):
            if c in ['@', '+']:
                return (x, y)
    return None

def _get_goals(level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    # ... (giữ nguyên mã của bạn)
    return {(x, y) for y, row in enumerate(level_data) for x, c in enumerate(row) if c in ['.', '+', '*']}

def _get_boxes(level_data: List[List[str]]) -> FrozenSet[Tuple[int, int]]:
    # ... (giữ nguyên mã của bạn)
    return frozenset((x, y) for y, row in enumerate(level_data) for x, c in enumerate(row) if c in ['$', '*'])

def _get_walls(level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    # ... (giữ nguyên mã của bạn)
    return {(x, y) for y, row in enumerate(level_data) for x, c in enumerate(row) if c == '#'}


# --- TỐI ƯU HÓA BẮT ĐẦU TỪ ĐÂY ---

def _precompute_deadlocks(walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]], level_data: List[List[str]]) -> Set[Tuple[int, int]]:
    """
    TÍNH TOÁN TRƯỚC các ô deadlock tĩnh. Một ô là deadlock nếu nó không phải là đích và:
    1. Bị kẹt ở góc.
    2. Bị kẹt dọc theo một bức tường mà không có lối thoát hoặc đích nào trên đường đi.
    """
    deadlocks = set()
    height = len(level_data)
    width = len(level_data[0]) if height > 0 else 0

    for r in range(height):
        for c in range(width):
            pos = (c, r)
            if pos in walls or pos in goals:
                continue

            # 1. Kiểm tra kẹt ở góc (giữa 2 bức tường)
            is_corner = ((c - 1, r) in walls and (c, r - 1) in walls) or \
                        ((c + 1, r) in walls and (c, r - 1) in walls) or \
                        ((c - 1, r) in walls and (c, r + 1) in walls) or \
                        ((c + 1, r) in walls and (c, r + 1) in walls)
            if is_corner:
                deadlocks.add(pos)
                continue

            # 2. Kiểm tra kẹt dọc tường (khó hơn)
            # Kẹt tường ngang (trên hoặc dưới)
            for dy in [-1, 1]:
                if (c, r + dy) in walls:
                    is_stuck = True
                    # Quét sang trái để tìm lối ra hoặc đích
                    for x_scan in range(c, -1, -1):
                        if (x_scan, r + dy) not in walls: is_stuck = False; break
                        if (x_scan, r) in goals: is_stuck = False; break
                    if not is_stuck: continue
                    
                    is_stuck = True
                    # Quét sang phải để tìm lối ra hoặc đích
                    for x_scan in range(c, width):
                        if (x_scan, r + dy) not in walls: is_stuck = False; break
                        if (x_scan, r) in goals: is_stuck = False; break
                    if is_stuck:
                        deadlocks.add(pos)

            # Kẹt tường dọc (trái hoặc phải)
            for dx in [-1, 1]:
                if (c + dx, r) in walls:
                    is_stuck = True
                    # Quét lên trên
                    for y_scan in range(r, -1, -1):
                        if (c + dx, y_scan) not in walls: is_stuck = False; break
                        if (c, y_scan) in goals: is_stuck = False; break
                    if not is_stuck: continue

                    is_stuck = True
                    # Quét xuống dưới
                    for y_scan in range(r, height):
                        if (c + dx, y_scan) not in walls: is_stuck = False; break
                        if (c, y_scan) in goals: is_stuck = False; break
                    if is_stuck:
                        deadlocks.add(pos)
                        
    return deadlocks

def energy_function(boxes: FrozenSet[Tuple[int, int]], goals: Set[Tuple[int, int]], deadlocks: Set[Tuple[int, int]]) -> int:
    """
    Hàm năng lượng được cải tiến:
    - Tổng khoảng cách Manhattan từ mỗi hộp đến đích gần nhất.
    - Phạt NẶNG nếu có bất kỳ hộp nào (chưa ở trên đích) nằm trong khu vực deadlock.
    """
    if not goals:
        return 0
    
    # Phạt nặng cho mỗi hộp trong khu vực deadlock
    for b in boxes:
        if b not in goals and b in deadlocks:
            # Trả về giá trị rất lớn để thuật toán gần như không bao giờ chọn trạng thái này
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
    
    # --- Khởi tạo ---
    walls = _get_walls(level_data)
    goals = _get_goals(level_data)
    # **TỐI ƯU**: Tính toán deadlock một lần duy nhất khi bắt đầu
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

    # --- Các hàm nội bộ (BFS và chuyển đổi tọa độ) không thay đổi ---
    # ... (giữ nguyên player_bfs và coords_to_actions)
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
    
    # **TỐI ƯU**: Truyền `deadlocks` vào hàm năng lượng
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
                # ... (phần xử lý timeout giữ nguyên)
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

            # Tạo các nước đi lân cận (chỉ đẩy hộp)
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

            # Chọn nước đi tiếp theo (giữ nguyên logic top-k)
            candidates.sort(key=lambda t: t[2])
            topk = candidates[:min(10, len(candidates))]
            actions_seq, next_state, next_energy = random.choice(topk)

            # Logic chấp nhận của Simulated Annealing
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
        
        # ... (phần restart và làm nhiễu giữ nguyên)

    if best_energy == 0:
        elapsed = time.time() - global_start
        save_sa_solution(level_idx, best_actions, elapsed)
        return best_actions

    print(f"SA: Không tìm thấy lời giải cho Level {level_idx}. Năng lượng tốt nhất={best_energy}")
    return None