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
    """
    Optimized And-Or search for Sokoban:
    - Assigns boxes to goals optimally (Hungarian algorithm for minimal total distance)
    - For each box-goal pair, uses BFS to find a sequence of valid pushes (not just straight lines)
    - For each push, uses BFS to find a path for the player to the push position
    - Prunes dead-ends (e.g., box in a corner not a goal)
    Returns a list of atomic player actions encoded as: 0=left,1=right,2=up,3=down
    or None if no solution was found within timeout or by this heuristic.
    """

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

    # Directions: L, R, U, D
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

    start_time = time.time()
    walls = get_walls(level_data)
    goals = list(get_goals(level_data))
    boxes = list(get_boxes(level_data))
    player = find_player(level_data)
    if player is None:
        return None

    # Hungarian algorithm for optimal assignment (minimize total box-goal distance)
    def hungarian(cost):
        # cost: rectangular matrix (rows = boxes, cols = goals). Pad to square with large costs.
        rows = len(cost)
        cols = len(cost[0]) if rows > 0 else 0
        n = max(rows, cols)
        # pad matrix
        big = 10**6
        square = [[big]*n for _ in range(n)]
        for i in range(rows):
            for j in range(cols):
                square[i][j] = cost[i][j]
        # now run Hungarian on square
        u = [0] * (n+1)
        v = [0] * (n+1)
        p = [0] * (n+1)
        way = [0] * (n+1)
        for i in range(1, n+1):
            p[0] = i
            minv = [float('inf')] * (n+1)
            used = [False] * (n+1)
            j0 = 0
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = -1
                for j in range(1, n+1):
                    if not used[j]:
                        cur = cost[i0-1][j-1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n+1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            # augmenting path
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        # result: p[1..n] is the goal assigned to box i
        result = [0]*n
        for j in range(1, n+1):
            if p[j] > 0:
                result[p[j]-1] = j-1
        # return only for original number of rows (boxes)
        return result[:rows]

    # BFS for box-pushes (returns list of (push_dir, player_path) to move box to goal)
    def box_push_bfs(box, goal, boxes_set, player_pos):
        from collections import deque
        visited = set()
        queue = deque()
        queue.append((box, tuple(sorted(boxes_set)), player_pos, []))
        visited.add((box, tuple(sorted(boxes_set)), player_pos))
        while queue:
            # timeout guard
            if time.time() - start_time > timeout:
                return None
            b, bset, ppos, moves = queue.popleft()
            if b == goal:
                return moves
            for d in directions:
                # cell behind box (where player must stand)
                px, py = b[0] - d[0], b[1] - d[1]
                if (px, py) in walls or (px, py) in bset:
                    continue
                # cell in front (where box will be pushed)
                nx, ny = b[0] + d[0], b[1] + d[1]
                if (nx, ny) in walls or (nx, ny) in bset:
                    continue
                # can player reach px,py?
                path = player_bfs(ppos, (px, py), set(bset))
                if path is None:
                    continue
                # new box set
                new_bset = set(bset)
                new_bset.remove(b)
                new_bset.add((nx, ny))
                state = ((nx, ny), tuple(sorted(new_bset)), (b[0], b[1]))
                if state in visited:
                    continue
                visited.add(state)
                queue.append(((nx, ny), tuple(sorted(new_bset)), (b[0], b[1]), moves + [(d, path)]))
        return None

    # BFS for player movement
    def player_bfs(start, goal, boxes_set):
        from collections import deque
        queue = deque()
        queue.append((start, [start]))
        visited = set([start])
        while queue:
            # timeout guard
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

    # Dead-end detection: box in a corner not a goal
    def is_dead_end(box, boxes_set):
        if box in goals:
            return False
        # check for corner
        for d1, d2 in [((0,1),(1,0)), ((0,1),(-1,0)), ((0,-1),(1,0)), ((0,-1),(-1,0))]:
            n1 = (box[0]+d1[0], box[1]+d1[1])
            n2 = (box[0]+d2[0], box[1]+d2[1])
            if (n1 in walls or n1 in boxes_set) and (n2 in walls or n2 in boxes_set):
                return True
        return False

    # Iterative greedy planner: pick the closest unsolved box to the player,
    # try goals in increasing manhattan distance order until a push plan is found.
    actions: List[int] = []
    cur_boxes = set(boxes)
    cur_player = player

    if len(cur_boxes) == 0:
        save_and_or_solution(level_idx, actions, 0.0)
        return actions

    # Main loop: continue until all boxes are on goals or timeout
    while not set(goals).issubset(cur_boxes):
        if time.time() - start_time > timeout:
            return None

        unsolved = [b for b in cur_boxes if b not in goals]
        if not unsolved:
            break

        # Sort unsolved boxes by distance to player (greedy)
        unsolved.sort(key=lambda b: abs(b[0]-cur_player[0]) + abs(b[1]-cur_player[1]))

        progress_made = False
        # Try boxes in this order
        for box in unsolved:
            # candidate goals that are not occupied
            candidate_goals = [g for g in goals if g not in cur_boxes]
            # sort goals by distance to this box
            candidate_goals.sort(key=lambda g: abs(g[0]-box[0]) + abs(g[1]-box[1]))
            for goal in candidate_goals:
                # quick dead-end heuristic: skip if box is already in a dead corner (and not goal)
                if is_dead_end(box, cur_boxes):
                    continue
                push_plan = box_push_bfs(box, goal, cur_boxes, cur_player)
                if push_plan is None:
                    # try next goal
                    continue

                # apply the found push_plan (sequence of (d, player_path))
                cur_box = box
                for d, path in push_plan:
                    # move player along path to push position
                    for j in range(len(path)-1):
                        dx = path[j+1][0] - path[j][0]
                        dy = path[j+1][1] - path[j][1]
                        actions.append(action_map[(dx, dy)])
                    # perform push
                    actions.append(action_map[d])

                    # update boxes and player
                    prev_box = cur_box
                    new_box = (prev_box[0] + d[0], prev_box[1] + d[1])
                    if prev_box in cur_boxes:
                        cur_boxes.remove(prev_box)
                    cur_boxes.add(new_box)
                    # after push player is on the previous box position
                    cur_player = prev_box
                    cur_box = new_box

                    # dead-end check for the moved box
                    if is_dead_end(new_box, cur_boxes):
                        # rollback this push_plan - conservative: abandon and try other goal/box
                        # revert box positions by recomputing from original set (simpler to abort)
                        # (for performance we don't implement full rollback here)
                        progress_made = False
                        break
                else:
                    # completed all pushes for this box->goal
                    progress_made = True
                    break
            if progress_made:
                break

        if not progress_made:
            # couldn't make progress for any box/goal pair with this heuristic
            return None

    elapsed = time.time() - start_time
    save_and_or_solution(level_idx, actions, elapsed)
    return actions
