import time
import heapq
from typing import List, Tuple, Optional, Set, FrozenSet


def save_and_or_solution(level_idx: int, path: Optional[List[int]], elapsed_time: float):
    if path is None:
        return
    try:
        with open("solutions.txt", "a", encoding="utf-8") as f:
            f.write(f"--- Level {level_idx} ---\n")
            f.write("--- Nhóm 4 ---\n")
            f.write(f"Thời gian chạy {elapsed_time:.10f} giây\n")
            f.write(f"Số bước And-Or: {len(path)}\n")
            f.write(f"Path And-Or: {path}\n\n")
    except Exception:
        pass


def solve_with_and_or_search(level_data: List[List[str]], level_idx: int, timeout: float = 10.0) -> Optional[List[int]]:
    def find_player(g: List[List[str]]) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(g):
            for x, c in enumerate(row):
                if c in ['@', '+']:
                    return (x, y)
        return None

    def get_goals(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, c in enumerate(row) if c in ['.', '+', '*']}

    def get_boxes(g: List[List[str]]) -> FrozenSet[Tuple[int, int]]:
        return frozenset((x, y) for y, row in enumerate(g) for x, c in enumerate(row) if c in ['$', '*'])

    def get_walls(g: List[List[str]]) -> Set[Tuple[int, int]]:
        return {(x, y) for y, row in enumerate(g) for x, c in enumerate(row) if c == '#'}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

    start_time = time.time()
    walls = get_walls(level_data)
    goals = list(get_goals(level_data))
    boxes = list(get_boxes(level_data))
    player = find_player(level_data)
    if player is None:
        return None

    def box_push_bfs(box, goal, boxes_set, player_pos):
        from collections import deque
        visited = set()
        queue = deque()
        queue.append((box, tuple(sorted(boxes_set)), player_pos, []))
        visited.add((box, tuple(sorted(boxes_set)), player_pos))
        while queue:
            if time.time() - start_time > timeout:
                return None
            b, bset, ppos, moves = queue.popleft()
            if b == goal:
                return moves
            for d in directions:
                px, py = b[0] - d[0], b[1] - d[1]
                if (px, py) in walls or (px, py) in bset:
                    continue
                nx, ny = b[0] + d[0], b[1] + d[1]
                if (nx, ny) in walls or (nx, ny) in bset:
                    continue
                path = player_bfs(ppos, (px, py), set(bset))
                if path is None:
                    continue
                new_bset = set(bset)
                new_bset.remove(b)
                new_bset.add((nx, ny))
                state = ((nx, ny), tuple(sorted(new_bset)), (b[0], b[1]))
                if state in visited:
                    continue
                visited.add(state)
                queue.append(((nx, ny), tuple(sorted(new_bset)), (b[0], b[1]), moves + [(d, path)]))
        return None

    def player_bfs(start, goal, boxes_set):
        from collections import deque
        queue = deque()
        queue.append((start, [start]))
        visited = set([start])
        while queue:
            if time.time() - start_time > timeout:
                return None
            pos, path = queue.popleft()
            if pos == goal:
                return path
            for d in directions:
                nx, ny = pos[0] + d[0], pos[1] + d[1]
                if (nx, ny) in walls or (nx, ny) in boxes_set or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
        return None

    def is_dead_end(box, boxes_set):
        if box in goals:
            return False

        for d1, d2 in [((0,1),(1,0)), ((0,1),(-1,0)), ((0,-1),(1,0)), ((0,-1),(-1,0))]:
            n1 = (box[0]+d1[0], box[1]+d1[1])
            n2 = (box[0]+d2[0], box[1]+d2[1])
            if (n1 in walls or n1 in boxes_set) and (n2 in walls or n2 in boxes_set):
                return True
        return False
    actions: List[int] = []
    cur_boxes = set(boxes)
    cur_player = player

    if len(cur_boxes) == 0:
        save_and_or_solution(level_idx, actions, 0.0)
        return actions

    while not set(goals).issubset(cur_boxes):
        if time.time() - start_time > timeout:
            return None

        unsolved = [b for b in cur_boxes if b not in goals]
        if not unsolved:
            break

        unsolved.sort(key=lambda b: abs(b[0]-cur_player[0]) + abs(b[1]-cur_player[1]))

        progress_made = False
        for box in unsolved:
            candidate_goals = [g for g in goals if g not in cur_boxes]
            candidate_goals.sort(key=lambda g: abs(g[0]-box[0]) + abs(g[1]-box[1]))
            for goal in candidate_goals:
                if is_dead_end(box, cur_boxes):
                    continue
                push_plan = box_push_bfs(box, goal, cur_boxes, cur_player)
                if push_plan is None:
                    continue

                cur_box = box
                for d, path in push_plan:
                    for j in range(len(path)-1):
                        dx = path[j+1][0] - path[j][0]
                        dy = path[j+1][1] - path[j][1]
                        actions.append(action_map[(dx, dy)])
                    actions.append(action_map[d])
                    prev_box = cur_box
                    new_box = (prev_box[0] + d[0], prev_box[1] + d[1])
                    if prev_box in cur_boxes:
                        cur_boxes.remove(prev_box)
                    cur_boxes.add(new_box)
                    cur_player = prev_box
                    cur_box = new_box

                    if is_dead_end(new_box, cur_boxes):
                        progress_made = False
                        break
                else:
                    progress_made = True
                    break
            if progress_made:
                break

        if not progress_made:
            return None

    elapsed = time.time() - start_time
    save_and_or_solution(level_idx, actions, elapsed)
    return actions
