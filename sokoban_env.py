import numpy as np
from sokoban import Game
from collections import deque

class SokobanEnv:
    def __init__(self, level=0, render=False):
        self.game = Game(level)
        self.num_levels = len(self.game.levels)
        self.action_space = 4
        self.observation_space = (8, 8)
        self.render_mode = render
        
    def reset(self):
        self.game.reset_level()
        if self.render_mode:
            self.game.draw()
        return self._get_state()
        
    def _get_state(self):
        state = np.zeros((8, 8), dtype=np.float32)
        for y in range(8):
            for x in range(8):
                cell = self.game.current_level[y][x]
                if cell == '#': state[y, x] = 1.0
                elif cell == '@': state[y, x] = 2.0
                elif cell == '$': state[y, x] = 3.0
                elif cell == '.': state[y, x] = 4.0
                elif cell == '*': state[y, x] = 5.0
                elif cell == '+': state[y, x] = 6.0
        return state.flatten()
        
    def _is_position_blocked(self, x, y):
        if not (0 <= x < 8 and 0 <= y < 8):
            return True
        cell = self.game.current_level[y][x]
        return cell in ['#', '$', '*']

    def _is_player_stuck(self):
        player_x, player_y = None, None
        for y in range(8):
            for x in range(8):
                if self.game.current_level[y][x] in ['@', '+']:
                    player_x, player_y = x, y
                    break
            if player_x is not None:
                break
        if player_x is None:
            return True
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        blocked_directions = sum(1 for dx, dy in directions if self._is_position_blocked(player_x + dx, player_y + dy))
        return blocked_directions == 4

    def _has_path_to_position(self, start_x, start_y, target_x, target_y):
        queue = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            if (x, y) == (target_x, target_y):
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and (nx, ny) not in visited:
                    if self.game.current_level[ny][nx] in [' ', '.', '@', '+']:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
        return False

    def _has_path_to_goal(self, box_x, box_y, goals):
        if not goals:
            return False
        for goal_x, goal_y in goals:
            if self._has_path_to_position(box_x, box_y, goal_x, goal_y):
                return True  # Bỏ kiểm tra người chơi để đơn giản hóa
        return False

    def _is_box_stuck(self, x, y):
        if self.game.current_level[y][x] not in ['$', '*']:
            return False
        # Chỉ kiểm tra nếu hộp bị chặn bởi tường ở hai phía đối diện
        for dx, dy in [(0, 1), (0, -1)]:
            if (0 <= x + dx < 8 and 0 <= y + dy < 8 and self.game.current_level[y + dy][x + dx] == '#'):
                opp_dx, opp_dy = -dx, -dy
                if (0 <= x + opp_dx < 8 and 0 <= y + opp_dy < 8 and 
                    self.game.current_level[y + opp_dy][x + opp_dx] in ['#', '$', '*']):
                    return True
        for dx, dy in [(1, 0), (-1, 0)]:
            if (0 <= x + dx < 8 and 0 <= y + dy < 8 and self.game.current_level[y + dy][x + dx] == '#'):
                opp_dx, opp_dy = -dx, -dy
                if (0 <= x + opp_dx < 8 and 0 <= y + opp_dy < 8 and 
                    self.game.current_level[y + opp_dy][x + opp_dx] in ['#', '$', '*']):
                    return True
        # Bỏ kiểm tra đường đi đến mục tiêu để giảm độ nghiêm ngặt
        return False

    def _check_stuck_boxes(self):
        for y in range(8):
            for x in range(8):
                if self.game.current_level[y][x] in ['$', '*']:
                    if self._is_box_stuck(x, y):
                        return True
        return False

    def _is_game_stuck(self):
        if self._is_player_stuck():
            return True, "player_stuck"
        if self._check_stuck_boxes():
            return True, "box_stuck"
        if len(self.game.history) > 20:
            current_state = str(self.game.current_level)
            recent_states = [str(state['level']) for state in self.game.history[-20:]]
            if recent_states.count(current_state) >= 5:
                return True, "state_loop"
        return False, ""

    def step(self, action):
        if action == 0: self.game.move(-1, 0)
        elif action == 1: self.game.move(1, 0)
        elif action == 2: self.game.move(0, -1)
        elif action == 3: self.game.move(0, 1)
            
        new_state = self._get_state()
        is_stuck, reason = self._is_game_stuck()
        reward = self._calculate_reward()
        done = self.game.is_complete()
        if is_stuck:
            reward -= 20.0  # Giảm hình phạt để khuyến khích khám phá
            print(f"Game stuck: {reason}, applying penalty")
        if self.render_mode:
            self.game.draw()
        return new_state, reward, done, {"stuck": is_stuck, "reason": reason}
        
    def _calculate_reward(self):
        reward = 0.0
        boxes_on_goals = sum(1 for y in range(8) for x in range(8) if self.game.current_level[y][x] == '*')
        total_goals = sum(1 for y in range(8) for x in range(8) if self.game.current_level[y][x] in ['.', '*', '+'])
        
        reward += boxes_on_goals * 15.0  # Tăng phần thưởng cho hộp trên mục tiêu
        if self.game.is_complete():
            reward += 300.0  # Tăng phần thưởng khi hoàn thành
        if self.game.box_pushes > 0:
            reward += 5.0  # Tăng phần thưởng cho việc đẩy hộp
        
        if len(self.game.history) > 1:
            prev_state = self.game.history[-2]['level']
            prev_boxes_on_goals = sum(1 for y in range(8) for x in range(8) if prev_state[y][x] == '*')
            if boxes_on_goals < prev_boxes_on_goals:
                reward -= 5.0  # Giảm hình phạt khi hộp rời mục tiêu
        
        boxes = [(x, y) for y in range(8) for x in range(8) if self.game.current_level[y][x] in ['$', '*']]
        goals = [(x, y) for y in range(8) for x in range(8) if self.game.current_level[y][x] in ['.', '*', '+']]
        if boxes and goals and len(self.game.history) > 1:
            prev_state = self.game.history[-2]['level']
            prev_boxes = [(x, y) for y in range(8) for x in range(8) if prev_state[y][x] in ['$', '*']]
            total_dist = sum(min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goals) for box in boxes)
            prev_total_dist = sum(min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goals) for box in prev_boxes)
            if total_dist < prev_total_dist:
                reward += 10.0  # Tăng phần thưởng khi tiến gần mục tiêu
        
        reward -= 0.001  # Giảm hình phạt mỗi bước
        if len(self.game.history) > 1 and str(self.game.current_level) == str(self.game.history[-2]['level']):
            reward -= 0.1  # Giảm hình phạt khi không tiến triển
        
        return reward
        
    def render(self):
        self.game.draw()