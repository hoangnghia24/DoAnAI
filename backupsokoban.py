import pygame
from game_constants import WALL, PLAYER, PLAYER_ON_GOAL, BOX, BOX_ON_GOAL, GOAL, FLOOR
import sys
import os
import heapq
from collections import deque
import numpy as np
import asyncio
import bfs_sokoban
import ast
import greedy_sokoban
import DLS_sokoban
import IDS_sokoban
import UCS_sokoban
import A_sokoban
import simulated_annealing_sokoban
import beam_search_sokoban
import genetic_algorithms_sokoban
import unobservable_sokoban
import backtracking_sokoban
import forward_checking_sokoban
import arc_consistency_sokoban
import and_or_search_sokoban
import hill_climbing_sokoban
#from game_core import Game
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 60
SCREEN_WIDTH = 10 * TILE_SIZE
SCREEN_HEIGHT = 10 * TILE_SIZE
MENU_WIDTH = 650
MENU_HEIGHT = 500
FPS = 165
# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
GOLD = (255, 215, 0)

LIGHT_SEA_GREEN = (173, 216, 230)

class SokobanEnv:
    def __init__(self, game):
        self.game = game
        self.action_space = 4
        self.observation_space = (8, 8)

    def reset(self):
        self.game.reset_level()
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

    def step(self, action):
        if action == 0: self.game.move(-1, 0)
        elif action == 1: self.game.move(1, 0)
        elif action == 2: self.game.move(0, -1)
        elif action == 3: self.game.move(0, 1)
        new_state = self._get_state()
        done = self.game.is_complete()
        return new_state, 0, done, {}

class Node:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.g = parent.g + 1 if parent else 0
        self.h = 0
        self.f = 0
        self.state = self._get_state()

    def _get_state(self):
        player_pos = None
        boxes = []
        for y in range(len(self.env.game.current_level)):
            for x in range(len(self.env.game.current_level[0])):
                cell = self.env.game.current_level[y][x]
                if cell in [PLAYER, PLAYER_ON_GOAL]:
                    player_pos = (x, y)
                if cell in [BOX, BOX_ON_GOAL]:
                    boxes.append((x, y))
        return (player_pos, tuple(sorted(boxes)))

    def F_Evaluation(self, heuristic=1):
        self.h = self._heuristic(heuristic)
        self.f = self.g + self.h

    def _heuristic(self, heuristic_type=1):
        player_pos, boxes = self.state
        goals = [(x, y) for y in range(len(self.env.game.current_level))
                 for x in range(len(self.env.game.current_level[0]))
                 if self.env.game.current_level[y][x] in [GOAL, PLAYER_ON_GOAL, BOX_ON_GOAL]]
        if not goals:
            return 0
        box_to_goal_dist = 0
        for box in boxes:
            min_distance = float('inf')
            for goal in goals:
                distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                min_distance = min(min_distance, distance)
            box_to_goal_dist += min_distance
        player_to_box_dist = float('inf')
        for box in boxes:
            distance = abs(player_pos[0] - box[0]) + abs(player_pos[1] - box[1])
            player_to_box_dist = min(player_to_box_dist, distance)
        return box_to_goal_dist + 0.5 * player_to_box_dist

    def succ(self):
        successors = deque()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        actions = [0, 1, 2, 3]
        player_pos, boxes = self.state
        boxes_set = set(boxes)

        for action, (dx, dy) in zip(actions, directions):
            new_x, new_y = player_pos[0] + dx, player_pos[1] + dy
            if not (0 <= new_x < len(self.env.game.current_level[0]) and
                    0 <= new_y < len(self.env.game.current_level)):
                continue
            if self.env.game.current_level[new_y][new_x] == WALL:
                continue
            if (new_x, new_y) in boxes_set:
                box_new_x, box_new_y = new_x + dx, new_y + dy
                if not (0 <= box_new_x < len(self.env.game.current_level[0]) and
                        0 <= box_new_y < len(self.env.game.current_level)):
                    continue
                if self.env.game.current_level[box_new_y][box_new_x] in [WALL, BOX, BOX_ON_GOAL]:
                    continue
                new_boxes = [(box_new_x, box_new_y) if box == (new_x, new_y) else box for box in boxes]
            else:
                new_boxes = boxes

            temp_env = SokobanEnv(self.env.game)
            temp_env.game.current_level = [row[:] for row in self.env.game.current_level]
            temp_env.game.player_pos = [player_pos[0], player_pos[1]]
            temp_env.game.history = []
            new_state, _, _, info = temp_env.step(action)

            child_env = SokobanEnv(self.env.game)
            child_env.game.current_level = [row[:] for row in temp_env.game.current_level]
            child_env.game.player_pos = [new_x, new_y]
            child_env.game.history = []

            child = Node(child_env, parent=self, action=action)
            successors.append(child)

        return successors

    def getSolution(self):
        solution = []
        current = self
        while current.parent:
            solution.append(current.action)
            current = current.parent
        return solution[::-1]

class Search:
    @staticmethod
    async def A(initial_node, heuristic=1, max_steps=10000):
        if initial_node.env.game.is_complete():
            return initial_node, 0

        initial_node.F_Evaluation(heuristic)
        open_list = [(initial_node.f, 0, initial_node)]
        heapq.heapify(open_list)
        closed_list = set()
        visited_states = set([initial_node.state])
        step = 0
        tiebreaker = 0

        while open_list and step < max_steps:
            step += 1
            _, _, current = heapq.heappop(open_list)
            closed_list.add(current.state)

            if current.env.game.is_complete():
                return current, step

            succ = current.succ()
            while succ:
                child = succ.popleft()
                if child.state in visited_states:
                    continue
                visited_states.add(child.state)
                child.F_Evaluation(heuristic)
                tiebreaker += 1
                heapq.heappush(open_list, (child.f, tiebreaker, child))
            await asyncio.sleep(0)

        return None, -1

class Game:
    def __init__(self, level=0, create_window=True):
        self.level = level
        self.screen = None
        if create_window:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH+200, SCREEN_HEIGHT))
            pygame.display.set_caption('Sokoban')
            self.clock = pygame.time.Clock()
        else:
            self.clock = None
        self.steps = 0
        self.box_pushes = 0
        self.history = []
        # Persistent dropdown state for algorithm selection (additive)
        # labels match the ones you already render in draw()
        self.dropdown_options = ['BFS solver', 'DLS solver','IDS solver','UCS solver','Greedy solver','A* solver','Simulated Annealing solver', 'Beam solver', 'Genetic solver'
                                 ,'And Or solver', 'Unobservable','Partially Observable solver','Backtracking solver','Forward Checking solver','Arc Consistency solver']
        self.dropdown_selected = 0
        self.dropdown_open = False
        self.dropdown_rect = pygame.Rect(SCREEN_WIDTH-100, 200, 250, 25)
        self.dropdown_item_height = 25
        # Flag to indicate a 'run all algorithms' operation is active
        self.running_all = False
        # Load images (assumes JPG folder exists)
        try:
            self.images = {
                'wall': pygame.image.load(os.path.join('JPG', 'Blocks', 'block_06.jpg')).convert_alpha(),
                'floor': pygame.image.load(os.path.join('JPG', 'Ground', 'ground.jpg')).convert_alpha(),
                'box': pygame.image.load(os.path.join('JPG', 'Crates', 'crate_05.jpg')).convert_alpha(),
                'box_on_goal': pygame.image.load(os.path.join('JPG', 'Crates', 'crate_05.jpg')).convert_alpha(),
                'player': pygame.image.load(os.path.join('JPG', 'playerFace.jpg')).convert_alpha(),
                'goal': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            }
            pygame.draw.rect(self.images['goal'], (135, 206, 235, 128), (0, 0, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(self.images['goal'], (0, 191, 255, 255), (0, 0, TILE_SIZE, TILE_SIZE), 2)
            for key in self.images:
                if key != 'goal':
                    self.images[key] = pygame.transform.scale(self.images[key], (TILE_SIZE, TILE_SIZE))
        except FileNotFoundError:
            # Fallback to placeholder surfaces if images not found
            self.images = {
                'wall': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA),
                'floor': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA),
                'box': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA),
                'box_on_goal': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA),
                'player': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA),
                'goal': pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            }
            self.images['wall'].fill((100, 100, 100))
            self.images['floor'].fill((200, 200, 200))
            self.images['box'].fill((139, 69, 19))
            self.images['box_on_goal'].fill((139, 69, 19))
            self.images['player'].fill((255, 0, 0))
            pygame.draw.rect(self.images['goal'], (135, 206, 235, 128), (0, 0, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(self.images['goal'], (0, 191, 255, 255), (0, 0, TILE_SIZE, TILE_SIZE), 2)

        self.levels = [
            # Level 0
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", " ", " ", " ", ".", " ", "#"],
             ["#", "#", " ", "$", " ", " ", " ", "#"],
             ["#", "#", " ", ".", " ", "$", ".", "#"],
             ["#", "#", " ", " ", " ", "$", " ", "#"],
             ["#", "#", " ", " ", "@", " ", " ", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 1
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", " ", " ", ".", ".", " ", "#"],
             ["#", "#", " ", " ", "$", " ", " ", "#"],
             ["#", "#", " ", " ", " ", "$", " ", "#"],
             ["#", "#", " ", " ", " ", " ", " ", "#"],
             ["#", "#", "#", "@", " ", " ", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 2
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", " ", "@", "#", "#"],
             ["#", " ", " ", ".", " ", " ", "#", "#"],
             ["#", " ", " ", " ", " ", " ", " ", "#"],
             ["#", "#", " ", " ", " ", " ", "#", "#"],
             ["#", "#", "$", " ", " ", " ", "#", "#"],
             ["#", "#", " ", ".", "$", " ", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 3
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", ".", " ", " ", " ", " ", "#"],
             ["#", "#", " ", " ", " ", " ", ".", "#"],
             ["#", "#", " ", "$", " ", " ", "@", "#"],
             ["#", "#", " ", " ", "$", " ", " ", "#"],
             ["#", "#", " ", " ", " ", " ", " ", "#"],
             ["#", "#", " ", " ", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 4
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", " ", " ", ".", "#", "#"],
             ["#", " ", " ", " ", " ", " ", "#", "#"],
             ["#", " ", "$", "$", " ", " ", "#", "#"],
             ["#", " ", ".", " ", "@", " ", "#", "#"],
             ["#", " ", " ", " ", " ", " ", "#", "#"],
             ["#", " ", " ", " ", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 5
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", " ", ".", ".", " ", " ", "#", "#"],
             ["#", " ", "$", " ", " ", " ", " ", "#"],
             ["#", " ", " ", " ", "$", " ", " ", "#"],
             ["#", " ", "@", " ", ".", "$", " ", "#"],
             ["#", "#", " ", " ", " ", " ", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 6
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", ".", " ", " ", "@", ".", "#", "#"],
             ["#", " ", "$", " ", " ", " ", "#", "#"],
             ["#", " ", " ", " ", "$", " ", " ", "#"],
             ["#", " ", " ", " ", " ", "#", " ", "#"],
             ["#", "#", "#", "#", " ", " ", " ", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 7
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", ".", " ", " ", "@", " ", "#", "#"],
             ["#", " ", "$", " ", " ", " ", "#", "#"],
             ["#", " ", " ", " ", "$", " ", " ", "#"],
             ["#", " ", " ", " ", " ", " ", " ", "#"],
             ["#", "#", "#", "#", " ", " ", ".", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 8
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", ".", " ", " ", "#", "#", "#", "#"],
             ["#", " ", " ", "$", " ", " ", "#", "#"],
             ["#", " ", " ", " ", "@", " ", "#", "#"],
             ["#", " ", "$", " ", " ", " ", "#", "#"],
             ["#", "#", " ", " ", ".", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 9
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", ".", " ", " ", "#", "#", "#", "#"],
             ["#", " ", " ", " ", " ", " ", "#", "#"],
             ["#", " ", ".", " ", " ", " ", "#", "#"],
             ["#", " ", "$", " ", "$", " ", "#", "#"],
             ["#", "#", "@", " ", " ", " ", "#", "#"],
             ["#", "#", " ", " ", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]],
            # Level 10
            [["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"],
             ["#", "#", " ", " ", " ", ".", "#", "#"],
             ["#", "#", " ", " ", " ", " ", "#", "#"],
             ["#", "@", " ", "$", " ", " ", "#", "#"],
             ["#", " ", " ", ".", "$", " ", "#", "#"],
             ["#", "#", "#", "#", "#", "#", "#", "#"]]
        ]
        self.current_level = self.levels[level]
        self.player_pos = self.find_player()
        self.save_state()

    def save_state(self):
        state = {
            'level': [row[:] for row in self.current_level],
            'player_pos': self.player_pos.copy() if self.player_pos else None,
            'steps': self.steps,
            'box_pushes': self.box_pushes
        }
        self.history.append(state)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            previous_state = self.history[-1]
            self.current_level = [row[:] for row in previous_state['level']]
            self.player_pos = previous_state['player_pos'].copy() if previous_state['player_pos'] else None
            self.steps = previous_state['steps']
            self.box_pushes = previous_state['box_pushes']
            return True
        return False

    def reset_level(self):
        if self.history:
            initial_state = self.history[0]
            self.current_level = [row[:] for row in initial_state['level']]
            self.player_pos = initial_state['player_pos'].copy() if initial_state['player_pos'] else None
            self.steps = initial_state['steps']
            self.box_pushes = initial_state['box_pushes']
            self.history = [initial_state]

    def find_player(self):
        for y, row in enumerate(self.current_level):
            for x, cell in enumerate(row):
                if cell in [PLAYER, PLAYER_ON_GOAL]:
                    return [x, y]
        return None

    def move(self, dx, dy):
        if not self.player_pos:
            return
        x, y = self.player_pos
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < 8 and 0 <= new_y < 8):
            return
        current = self.current_level[y][x]
        next_cell = self.current_level[new_y][new_x]
        if next_cell == WALL:
            return
        if next_cell in [BOX, BOX_ON_GOAL]:
            box_x, box_y = new_x + dx, new_y + dy
            if not (0 <= box_x < 8 and 0 <= box_y < 8):
                return
            if self.current_level[box_y][box_x] in [WALL, BOX, BOX_ON_GOAL]:
                return
            if self.current_level[box_y][box_x] == GOAL:
                self.current_level[box_y][box_x] = BOX_ON_GOAL
            else:
                self.current_level[box_y][box_x] = BOX
            self.box_pushes += 1
            if next_cell == BOX_ON_GOAL:
                next_cell = GOAL
            else:
                next_cell = FLOOR
        if next_cell == GOAL:
            self.current_level[new_y][new_x] = PLAYER_ON_GOAL
        else:
            self.current_level[new_y][new_x] = PLAYER
        if current == PLAYER_ON_GOAL:
            self.current_level[y][x] = GOAL
        else:
            self.current_level[y][x] = FLOOR
        self.player_pos = [new_x, new_y]
        self.steps += 1
        self.save_state()

    def is_complete(self):
        goals = 0
        boxes_on_goals = 0
        for row in self.current_level:
            for cell in row:
                if cell in [GOAL, PLAYER_ON_GOAL]:
                    goals += 1
                if cell == BOX_ON_GOAL:
                    boxes_on_goals += 1
                    goals += 1
        return boxes_on_goals == goals and goals > 0

    def draw(self):
        self.screen.fill(BLACK)
        for y, row in enumerate(self.current_level):
            for x, cell in enumerate(row):
                pos = (x * TILE_SIZE, y * TILE_SIZE)
                self.screen.blit(self.images['floor'], pos)
                if cell in [GOAL, PLAYER_ON_GOAL, BOX_ON_GOAL]:
                    self.screen.blit(self.images['goal'], pos)
                if cell == WALL:
                    self.screen.blit(self.images['wall'], pos)
                if cell == BOX:
                    self.screen.blit(self.images['box'], pos)
                elif cell == BOX_ON_GOAL:
                    self.screen.blit(self.images['box_on_goal'], pos)
                if cell in [PLAYER, PLAYER_ON_GOAL]:
                    self.screen.blit(self.images['player'], pos)
        font = pygame.font.Font(None, 36)
        steps_text = font.render(f'Steps: {self.steps}', True, WHITE)
        pushes_text = font.render(f'Pushes: {self.box_pushes}', True, WHITE)
        level_text = font.render(f'Level: {self.level}', True, WHITE)
        self.screen.blit(steps_text, (10, SCREEN_HEIGHT - 90))
        self.screen.blit(pushes_text, (10, SCREEN_HEIGHT - 60))
        self.screen.blit(level_text, (10, SCREEN_HEIGHT - 30))
        control_font = pygame.font.Font(None, 24)
        menu_control = ['Back to menu','Reset level', 'Undo','Next level','Run all algorithms']
        r_text = control_font.render('Reset level', True, WHITE)
        u_text = control_font.render('Undo', True, WHITE)
        m_text = control_font.render('Back to menu', True, WHITE)
        n_text = control_font.render('Next level', True, WHITE)
        run_all_text = control_font.render('Run all algorithms', True, WHITE)
        # Draw dropdown header (shows current selection)
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.rect(self.screen, DARK_GRAY, self.dropdown_rect)
        sel_label = self.dropdown_options[self.dropdown_selected]
        sel_text = control_font.render(sel_label, True, WHITE)
        sel_rect = sel_text.get_rect(center=self.dropdown_rect.center)
        self.screen.blit(sel_text, sel_rect)
        # If open, draw options beneath the header
        if self.dropdown_open:
            for i, option in enumerate(self.dropdown_options):
                option_rect = pygame.Rect(self.dropdown_rect.x, self.dropdown_rect.y + (i+1)*self.dropdown_item_height, self.dropdown_rect.width, self.dropdown_item_height)
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, LIGHT_SEA_GREEN, option_rect)
                else:
                    pygame.draw.rect(self.screen, DARK_GRAY, option_rect)
                option_text = control_font.render(option, True, WHITE)
                option_text_rect = option_text.get_rect(center=option_rect.center)
                self.screen.blit(option_text, option_text_rect)
        buttons = []
        for i, option in enumerate(menu_control):
            button_rect =  pygame.Rect(SCREEN_HEIGHT-100, 10 + i*30, 200, 25)
            if button_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(self.screen, LIGHT_SEA_GREEN, button_rect)
            else:
                pygame.draw.rect(self.screen, DARK_GRAY, button_rect)
            text = [m_text, r_text, u_text, n_text,run_all_text][i]
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            buttons.append(button_rect)
        pygame.display.flip()
        return buttons

    async def flash_message(self, text_str, duration=0.5, font_size=36):
        """Show a centered transient message for `duration` seconds then redraw.

        This replaces ad-hoc blit/flip/sleep blocks so messages don't stack
        on-screen during rapid batch runs.
        """
        font = pygame.font.Font(None, font_size)
        text = font.render(text_str, True, WHITE)
        text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        # draw over current frame
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        await asyncio.sleep(duration)
        # redraw the game contents to remove the message
        self.draw()

    async def run_greedy(self):
        """Run BFS solver in a threadpool and animate the resulting path.

        This keeps the UI responsive by delegating the CPU-bound solver to an
        executor, then animates any returned path on the main thread. If the
        solver doesn't return a path, try to parse the last saved path for the
        level from bfs_solutions.txt as a fallback.
        """
        await self.flash_message('Greedy start', duration=0.5, font_size=74)
        loop = asyncio.get_running_loop()
        level_copy = [row[:] for row in self.current_level]
        try:
            print(f"Greedy: Starting solver for level {self.level} (executor)")
            path = await loop.run_in_executor(None, greedy_sokoban.solve_with_greedy, level_copy, self.level)
            print(f"Greedy: Solver returned: {path}")
        except Exception as e:
            print(f"Error running Greedy in executor: {e}")
            path = None
        if not path:
            try:
                with open('solutions.txt', 'r', encoding='utf-8') as f:
                    data = f.read()
                marker = f"--- Level {self.level} ---"
                if marker in data:
                    parts = data.split(marker)
                    after = parts[-1]
                    if 'Path Greedy:' in after:
                        path_line = after.split('Path Greedy:', 1)[1].strip().split('\n', 1)[0]
                        try:
                            path = ast.literal_eval(path_line)
                            print(f"Greedy: Loaded path from solutions.txt for level {self.level}: {path}")
                        except Exception as e:
                            print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error parsing solutions.txt: {e}")

        if not path:
            await self.flash_message('No Greedy solution found', duration=1.5, font_size=36)
            return

        # Animate the path
        self.reset_level()
        self.draw()
        await asyncio.sleep(0.2)

        for action in path:
            if action == 0:
                self.move(-1, 0)
            elif action == 1:
                self.move(1, 0)
            elif action == 2:
                self.move(0, -1)
            elif action == 3:
                self.move(0, 1)
            self.draw()
            await asyncio.sleep(0.2)

    async def run_and_or_search(self):
        """Run BFS solver in a threadpool and animate the resulting path.

        This keeps the UI responsive by delegating the CPU-bound solver to an
        executor, then animates any returned path on the main thread. If the
        solver doesn't return a path, try to parse the last saved path for the
        level from bfs_solutions.txt as a fallback.
        """
        await self.flash_message('And-Or start', duration=0.5, font_size=74)
        loop = asyncio.get_running_loop()
        level_copy = [row[:] for row in self.current_level]
        try:
            print(f"And-Or: Starting solver for level {self.level} (executor)")
            path = await loop.run_in_executor(None, and_or_search_sokoban.solve_with_and_or_search, level_copy, self.level)
            print(f"And-Or: Solver returned: {path}")
        except Exception as e:
            print(f"Error running And-Or in executor: {e}")
            path = None
        if not path:
            try:
                with open('solutions.txt', 'r', encoding='utf-8') as f:
                    data = f.read()
                marker = f"--- Level {self.level} ---"
                if marker in data:
                    parts = data.split(marker)
                    after = parts[-1]
                    if 'Path And-Or:' in after:
                        path_line = after.split('Path And-Or:', 1)[1].strip().split('\n', 1)[0]
                        try:
                            path = ast.literal_eval(path_line)
                            print(f"And-Or: Loaded path from solutions.txt for level {self.level}: {path}")
                        except Exception as e:
                            print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error parsing solutions.txt: {e}")

        if not path:
            await self.flash_message('No And-Or solution found', duration=1.5, font_size=36)
            return

        # Animate the path
        self.reset_level()
        self.draw()
        await asyncio.sleep(0.2)

        for action in path:
            if action == 0:
                self.move(-1, 0)
            elif action == 1:
                self.move(1, 0)
            elif action == 2:
                self.move(0, -1)
            elif action == 3:
                self.move(0, 1)
            self.draw()
            await asyncio.sleep(0.2)

    async def run_forward(self):
        """Run BFS solver in a threadpool and animate the resulting path.

        This keeps the UI responsive by delegating the CPU-bound solver to an
        executor, then animates any returned path on the main thread. If the
        solver doesn't return a path, try to parse the last saved path for the
        level from bfs_solutions.txt as a fallback.
        """
        await self.flash_message('Forward Checking start', duration=0.5, font_size=74)
        loop = asyncio.get_running_loop()
        level_copy = [row[:] for row in self.current_level]
        try:
            print(f"Forward Checking: Starting solver for level {self.level} (executor)")
            path = await loop.run_in_executor(None, forward_checking_sokoban.solve_with_forward_checking, level_copy, self.level)
            print(f"Forward Checking: Solver returned: {path}")
        except Exception as e:
            print(f"Error running Forward Checking in executor: {e}")
            path = None
        if not path:
            try:
                with open('solutions.txt', 'r', encoding='utf-8') as f:
                    data = f.read()
                marker = f"--- Level {self.level} ---"
                if marker in data:
                    parts = data.split(marker)
                    after = parts[-1]
                    if 'Path Forward Checking:' in after:
                        path_line = after.split('Path Forward Checking:', 1)[1].strip().split('\n', 1)[0]
                        try:
                            path = ast.literal_eval(path_line)
                            print(f"Forward Checking: Loaded path from solutions.txt for level {self.level}: {path}")
                        except Exception as e:
                            print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error parsing solutions.txt: {e}")

        if not path:
            await self.flash_message('No Forward Checking solution found', duration=1.5, font_size=36)
            return

        # Animate the path
        self.reset_level()
        self.draw()
        await asyncio.sleep(0.2)

        for action in path:
            if action == 0:
                self.move(-1, 0)
            elif action == 1:
                self.move(1, 0)
            elif action == 2:
                self.move(0, -1)
            elif action == 3:
                self.move(0, 1)
            self.draw()
            await asyncio.sleep(0.2)

    async def run_arc(self):
            """Run BFS solver in a threadpool and animate the resulting path.

            This keeps the UI responsive by delegating the CPU-bound solver to an
            executor, then animates any returned path on the main thread. If the
            solver doesn't return a path, try to parse the last saved path for the
            level from bfs_solutions.txt as a fallback.
            """
            await self.flash_message('Arc Consistency start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"Arc Consistency: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, arc_consistency_sokoban.solve_with_arc_consistency, level_copy, self.level)
                print(f"Arc Consistency: Solver returned: {path}")
            except Exception as e:
                print(f"Error running Arc Consistency in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path Arc Consistency:' in after:
                            path_line = after.split('Path Forward Checking:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"Arc Consistency: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No Arc Consistency solution found', duration=1.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_bfs(self):
        """Run BFS solver in a threadpool and animate the resulting path.

        This keeps the UI responsive by delegating the CPU-bound solver to an
        executor, then animates any returned path on the main thread. If the
        solver doesn't return a path, try to parse the last saved path for the
        level from bfs_solutions.txt as a fallback.
        """
        # If a 'run all' operation is active, wait until it finishes so
        # individual BFS doesn't start concurrently with the batch run.

        await self.flash_message('BFS start', duration=0.5, font_size=74)
        loop = asyncio.get_running_loop()
        level_copy = [row[:] for row in self.current_level]
        try:
            print(f"BFS: Starting solver for level {self.level} (executor)")
            path = await loop.run_in_executor(None, bfs_sokoban.solve_with_bfs, level_copy, self.level)
            print(f"BFS: Solver returned: {path}")
        except Exception as e:
            print(f"Error running BFS in executor: {e}")
            path = None
        if not path:
            try:
                with open('solutions.txt', 'r', encoding='utf-8') as f:
                    data = f.read()
                marker = f"--- Level {self.level} ---"
                if marker in data:
                    parts = data.split(marker)
                    after = parts[-1]
                    if 'Path BFS:' in after:
                        path_line = after.split('Path BFS:', 1)[1].strip().split('\n', 1)[0]
                        try:
                            path = ast.literal_eval(path_line)
                            print(f"BFS: Loaded path from solutions.txt for level {self.level}: {path}")
                        except Exception as e:
                            print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error parsing bfs_solutions.txt: {e}")

        if not path:
            await self.flash_message('No BFS solution found', duration=1.0, font_size=36)
            return

        # Animate the path
        self.reset_level()
        self.draw()
        await asyncio.sleep(0.2)

        for action in path:
            if action == 0:
                self.move(-1, 0)
            elif action == 1:
                self.move(1, 0)
            elif action == 2:
                self.move(0, -1)
            elif action == 3:
                self.move(0, 1)
            self.draw()
            await asyncio.sleep(0.2)

    async def run_dls(self):
        await self.flash_message('DLS start', duration=0.5, font_size=74)
        loop = asyncio.get_running_loop()
        level_copy = [row[:] for row in self.current_level]
        try:
            print(f"DLS: Starting solver for level {self.level} (executor)")
            path = await loop.run_in_executor(None, DLS_sokoban.solve_with_dls, level_copy, self.level)
            print(f"DLS: Solver returned: {path}")
        except Exception as e:
            print(f"Error running DLS in executor: {e}")
            path = None
        if not path:
            try:
                with open('solutions.txt', 'r', encoding='utf-8') as f:
                    data = f.read()
                marker = f"--- Level {self.level} ---"
                if marker in data:
                    parts = data.split(marker)
                    after = parts[-1]
                    if 'Path DLS:' in after:
                        path_line = after.split('Path DLS:', 1)[1].strip().split('\n', 1)[0]
                        try:
                            path = ast.literal_eval(path_line)
                            print(f"DLS: Loaded path from solutions.txt for level {self.level}: {path}")
                        except Exception as e:
                            print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error parsing solutions.txt: {e}")

        if not path:
            await self.flash_message('No DLS solution found', duration=0.5, font_size=36)
            return

        # Animate the path
        self.reset_level()
        self.draw()
        await asyncio.sleep(0.2)

        for action in path:
            if action == 0:
                self.move(-1, 0)
            elif action == 1:
                self.move(1, 0)
            elif action == 2:
                self.move(0, -1)
            elif action == 3:
                self.move(0, 1)
            self.draw()
            await asyncio.sleep(0.2)

    async def run_ids(self):
            await self.flash_message('IDS start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"IDS: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, IDS_sokoban.solve_with_ids, level_copy, self.level)
                print(f"IDS: Solver returned: {path}")
            except Exception as e:
                print(f"Error running BFS in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path IDS:' in after:
                            path_line = after.split('Path IDS:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"BFS: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No IDS solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_ucs(self):
            await self.flash_message('UCS start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"UCS: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, UCS_sokoban.solve_with_ucs, level_copy, self.level)
                print(f"UCS: Solver returned: {path}")
            except Exception as e:
                print(f"Error running UCS in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path UCS:' in after:
                            path_line = after.split('Path UCS:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"UCS: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No UCS solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_a(self):
        await self.flash_message('A* start', duration=0.5, font_size=74)
        loop = asyncio.get_running_loop()
        level_copy = [row[:] for row in self.current_level]
        try:
            print(f"A*: Starting solver for level {self.level} (executor)")
            path = await loop.run_in_executor(None, A_sokoban.solve_with_a_star, level_copy, self.level)
            print(f"A*: Solver returned: {path}")
        except Exception as e:
            print(f"Error running A* in executor: {e}")
            path = None
        if not path:
            try:
                with open('solutions.txt', 'r', encoding='utf-8') as f:
                    data = f.read()
                marker = f"--- Level {self.level} ---"
                if marker in data:
                    parts = data.split(marker)
                    after = parts[-1]
                    if 'Path A*:' in after:
                        path_line = after.split('Path A*:', 1)[1].strip().split('\n', 1)[0]
                        try:
                            path = ast.literal_eval(path_line)
                            print(f"A*: Loaded path from solutions.txt for level {self.level}: {path}")
                        except Exception as e:
                            print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error parsing solutions.txt: {e}")

        if not path:
            await self.flash_message('No A* solution found', duration=0.5, font_size=36)
            return

        # Animate the path
        self.reset_level()
        self.draw()
        await asyncio.sleep(0.2)

        for action in path:
            if action == 0:
                self.move(-1, 0)
            elif action == 1:
                self.move(1, 0)
            elif action == 2:
                self.move(0, -1)
            elif action == 3:
                self.move(0, 1)
            self.draw()
            await asyncio.sleep(0.2)

    async def run_simulated_annealing(self):
            await self.flash_message('Simulated annealing start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"SA: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, simulated_annealing_sokoban.solve_with_simulated_annealing, level_copy, self.level)
                print(f"SA: Solver returned: {path}")
            except Exception as e:
                print(f"Error running SA in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path SA:' in after:
                            path_line = after.split('Path SA:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"SA: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No SA solution found', duration=1.0, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_beam(self):
            await self.flash_message('Beam start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"Beam: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, beam_search_sokoban.solve_with_beam_search, level_copy, self.level)
                print(f"Beam: Solver returned: {path}")
            except Exception as e:
                print(f"Error running Beam in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path Beam:' in after:
                            path_line = after.split('Path Beam:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"Beam: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No Beam solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_genetic(self):
            await self.flash_message('Genetic start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"Genetic: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, genetic_algorithms_sokoban.solve_with_genetic_algorithm, level_copy, self.level)
                print(f"Genetic: Solver returned: {path}")
            except Exception as e:
                print(f"Error running Genetic in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path Genetic:' in after:
                            path_line = after.split('Path Genetic:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"Genetic: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No Genetic solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_unobservable(self):
            await self.flash_message('Unobservable start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"Unobservable: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, unobservable_sokoban.solve_with_unobservable_search, level_copy, self.level)
                print(f"Unobservable: Solver returned: {path}")
            except Exception as e:
                print(f"Error running Unobservable in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path Unobservable:' in after:
                            path_line = after.split('Path Unobservable:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"Unobservable: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No Unobservable solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_backtracking(self):
            await self.flash_message('Backtracking start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"Backtracking: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, backtracking_sokoban.solve_with_backtracking, level_copy, self.level)
                print(f"Backtracking: Solver returned: {path}")
            except Exception as e:
                print(f"Error running Backtracking in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path Backtracking:' in after:
                            path_line = after.split('Path Backtracking:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"Backtracking: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No Backtracking solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_partially_observable(self):
            await self.flash_message('Partially Observable start', duration=0.5, font_size=74)
            loop = asyncio.get_running_loop()
            level_copy = [row[:] for row in self.current_level]
            try:
                print(f"Partially Observable: Starting solver for level {self.level} (executor)")
                path = await loop.run_in_executor(None, backtracking_sokoban.solve_with_backtracking, level_copy, self.level)
                print(f"Partially Observable: Solver returned: {path}")
            except Exception as e:
                print(f"Error running Partially Observable in executor: {e}")
                path = None
            if not path:
                try:
                    with open('solutions.txt', 'r', encoding='utf-8') as f:
                        data = f.read()
                    marker = f"--- Level {self.level} ---"
                    if marker in data:
                        parts = data.split(marker)
                        after = parts[-1]
                        if 'Path Partially Observable:' in after:
                            path_line = after.split('Path Partially Observable:', 1)[1].strip().split('\n', 1)[0]
                            try:
                                path = ast.literal_eval(path_line)
                                print(f"Partially Observable: Loaded path from solutions.txt for level {self.level}: {path}")
                            except Exception as e:
                                print(f"Error parsing path line from solutions.txt: {e}; raw={path_line!r}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error parsing solutions.txt: {e}")

            if not path:
                await self.flash_message('No Partially Observable solution found', duration=0.5, font_size=36)
                return

            # Animate the path
            self.reset_level()
            self.draw()
            await asyncio.sleep(0.2)

            for action in path:
                if action == 0:
                    self.move(-1, 0)
                elif action == 1:
                    self.move(1, 0)
                elif action == 2:
                    self.move(0, -1)
                elif action == 3:
                    self.move(0, 1)
                self.draw()
                await asyncio.sleep(0.2)

    async def run_all_algorithms(self):
        self.reset_level()
        await self.run_bfs()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_dls()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_ids()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_ucs()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_greedy()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_a()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_simulated_annealing()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_beam()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_genetic()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_and_or_search()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_unobservable()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_unobservable()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_backtracking()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_forward()
        await asyncio.sleep(1)

        self.reset_level()
        await self.run_arc()

    async def run(self):
        running = True
        buttons_control = self.draw()
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move(1, 0)
                    elif event.key == pygame.K_UP:
                        self.move(0, -1)
                    elif event.key == pygame.K_DOWN:
                        self.move(0, 1)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if event.button == 1:
                        if self.dropdown_rect.collidepoint(mouse_pos):
                            self.dropdown_open = not self.dropdown_open
                        elif self.dropdown_open:
                            for idx in range(len(self.dropdown_options)):
                                item_rect = pygame.Rect(self.dropdown_rect.x, self.dropdown_rect.y + (idx+1)*self.dropdown_item_height, self.dropdown_rect.width, self.dropdown_item_height)
                                if item_rect.collidepoint(mouse_pos):
                                    self.dropdown_selected = idx
                                    self.dropdown_open = False
                                    try:
                                        if idx == 0:
                                            await self.run_bfs()
                                        elif idx == 1:
                                            await self.run_dls()
                                        elif idx == 2:
                                            await self.run_ids()
                                        elif idx == 3:
                                            await self.run_ucs()
                                        elif idx == 4:
                                            await self.run_greedy()
                                        elif idx == 5:
                                            await self.run_a()
                                        elif idx == 6:
                                            await self.run_simulated_annealing()
                                        elif idx == 7:
                                            await self.run_beam()
                                        elif idx == 8:
                                            await self.run_genetic()
                                        elif idx == 9:
                                            await self.run_and_or_search()
                                        elif idx == 10:
                                            await self.run_unobservable()
                                        elif idx == 11:
                                            await self.run_partially_observable()
                                        elif idx == 12:
                                            await self.run_backtracking()
                                        elif idx == 13:
                                            await self.run_forward()
                                        elif idx == 14:
                                            await self.run_arc()
                                    except Exception as e:
                                        print(f"Error running selected algorithm: {e}")
                                    break
                    for i, button in enumerate(buttons_control):
                        if button.collidepoint(pygame.mouse.get_pos()):
                            select_option = i
                            if select_option == 0:
                                return "menu"
                            elif select_option == 1:
                                self.reset_level()
                            elif select_option == 2:
                                self.undo()
                            elif select_option == 3 and self.is_complete():
                                if self.level + 1 < len(self.levels):
                                    self.__init__(self.level + 1)
                                else:
                                    return "menu"
                            elif select_option == 4:
                                await self.run_all_algorithms()
            self.draw()
            if self.is_complete():
                font = pygame.font.Font(None, 74)
                text = font.render('Level Complete!', True, WHITE)
                text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                self.screen.blit(text, text_rect)
                pygame.display.flip()
                await asyncio.sleep(1)
        return "menu"

def draw_menu(screen, selected_option, mouse_pos):
    screen.fill(LIGHT_SEA_GREEN)
    title_font = pygame.font.Font(None, 60)
    title_text = title_font.render('Sokoban', True, GOLD)
    title_shadow = title_font.render('Sokoban', True, BLACK)
    title_rect = title_text.get_rect(center=(MENU_WIDTH // 2, 80))
    screen.blit(title_shadow, (title_rect.x + 2, title_rect.y + 2))
    screen.blit(title_text, title_rect)
    menu_options = ['Start', 'Choose level', 'Exit']
    font = pygame.font.Font(None, 40)
    buttons = []
    for i, option in enumerate(menu_options):
        button_rect = pygame.Rect(MENU_WIDTH  // 2 - 100, 150 + i * 80, 200, 50)
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, LIGHT_GRAY, button_rect)
            pygame.draw.rect(screen, GOLD, button_rect, 3)
        elif i == selected_option:
            pygame.draw.rect(screen, LIGHT_GRAY, button_rect)
            pygame.draw.rect(screen, GOLD, button_rect, 3)
        else:
            pygame.draw.rect(screen, WHITE, button_rect)
            pygame.draw.rect(screen, BLACK, button_rect, 2)
        text = font.render(option, True, BLACK)
        text_rect = text.get_rect(center=button_rect.center)
        screen.blit(text, text_rect)
        buttons.append(button_rect)
    pygame.display.flip()
    return buttons

def select_level(screen):
    font = pygame.font.Font(None, 40)
    levels = Game(create_window=False).levels
    total_levels = len(levels)
    selected_level = -1
    scroll_offset = 0
    max_display_levels = 5
    input_text = ""
    input_active = False
    running = True
    input_box = pygame.Rect(MENU_WIDTH // 2 - 50, MENU_HEIGHT // 2 - 100, 100, 40)
    confirm_button = pygame.Rect(MENU_WIDTH // 2 + 60, MENU_HEIGHT // 2 - 100, 100, 40)
    back_button = pygame.Rect(MENU_WIDTH // 2 - 160, MENU_HEIGHT // 2 - 100, 100, 40)
    list_box_width = 200
    list_box_height = max_display_levels * 40 + 10
    list_box = pygame.Rect(MENU_WIDTH // 2 - list_box_width // 2, MENU_HEIGHT // 2 - 20, list_box_width, list_box_height)
    while running:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(LIGHT_SEA_GREEN)
        title_font = pygame.font.Font(None, 60)
        title_text = title_font.render('Choose Level', True, GOLD)
        title_shadow = title_font.render('Choose Level', True, BLACK)
        title_rect = title_text.get_rect(center=(MENU_WIDTH // 2, MENU_HEIGHT // 2 - 150))
        screen.blit(title_shadow, (title_rect.x + 2, title_rect.y + 2))
        screen.blit(title_text, title_rect)
        label_text = font.render("Select Your Level:", True, BLACK)
        label_rect = label_text.get_rect(center=(MENU_WIDTH // 2, MENU_HEIGHT // 2 - 120))
        screen.blit(label_text, label_rect)
        pygame.draw.rect(screen, WHITE, input_box)
        pygame.draw.rect(screen, BLACK, input_box, 2)
        input_surface = font.render(input_text, True, BLACK)
        screen.blit(input_surface, (input_box.x + 5, input_box.y + 5))
        if confirm_button.collidepoint(mouse_pos):
            pygame.draw.rect(screen, LIGHT_GRAY, confirm_button)
            pygame.draw.rect(screen, GOLD, confirm_button, 2)
        else:
            pygame.draw.rect(screen, LIGHT_GRAY, confirm_button)
            pygame.draw.rect(screen, BLACK, confirm_button, 2)
        confirm_text = font.render("Go", True, BLACK)
        confirm_rect = confirm_text.get_rect(center=confirm_button.center)
        screen.blit(confirm_text, confirm_rect)
        if back_button.collidepoint(mouse_pos):
            pygame.draw.rect(screen, LIGHT_GRAY, back_button)
            pygame.draw.rect(screen, GOLD, back_button, 2)
        else:
            pygame.draw.rect(screen, LIGHT_GRAY, back_button)
            pygame.draw.rect(screen, BLACK, back_button, 2)
        back_text = font.render("Back", True, BLACK)
        back_rect = back_text.get_rect(center=back_button.center)
        screen.blit(back_text, back_rect)
        pygame.draw.rect(screen, WHITE, list_box)
        pygame.draw.rect(screen, BLACK, list_box, 2)
        level_buttons = []
        for i in range(scroll_offset, min(scroll_offset + max_display_levels, total_levels)):
            button_y = list_box.y + (i - scroll_offset) * 40 + 5
            button_rect = pygame.Rect(list_box.x + 5, button_y, list_box.width - 10, 35)
            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_GRAY, button_rect)
                pygame.draw.rect(screen, GOLD, button_rect, 2)
            elif i == selected_level:
                pygame.draw.rect(screen, LIGHT_GRAY, button_rect)
                pygame.draw.rect(screen, GOLD, button_rect, 2)
            else:
                pygame.draw.rect(screen, WHITE, button_rect)
                pygame.draw.rect(screen, BLACK, button_rect, 2)
            text = font.render(f"Level {i}", True, BLACK)
            text_rect = text.get_rect(center=button_rect.center)
            screen.blit(text, text_rect)
            level_buttons.append((button_rect, i))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif input_active:
                    if event.key == pygame.K_RETURN:
                        try:
                            level_num = int(input_text) - 1
                            if 0 <= level_num < total_levels:
                                return level_num
                        except ValueError:
                            pass
                        input_text = ""
                        input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        if event.unicode.isdigit():
                            input_text += event.unicode
                else:
                    if event.key == pygame.K_UP:
                        selected_level = max(0, selected_level - 1)
                        if selected_level < scroll_offset:
                            scroll_offset = selected_level
                    elif event.key == pygame.K_DOWN:
                        selected_level = min(total_levels - 1, selected_level + 1)
                        if selected_level >= scroll_offset + max_display_levels:
                            scroll_offset = selected_level - max_display_levels + 1
                    elif event.key == pygame.K_RETURN:
                        return selected_level
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if input_box.collidepoint(mouse_pos):
                    input_active = True
                elif confirm_button.collidepoint(mouse_pos) and input_text:
                    try:
                        level_num = int(input_text) - 1
                        if 0 <= level_num < total_levels:
                            return level_num
                    except ValueError:
                        pass
                    input_text = ""
                    input_active = False
                elif back_button.collidepoint(mouse_pos):
                    return None
                else:
                    input_active = False
                    for button_rect, level_idx in level_buttons:
                        if button_rect.collidepoint(mouse_pos) and event.button == 1:
                            return level_idx
            elif event.type == pygame.MOUSEWHEEL:
                scroll_offset = max(0, min(scroll_offset - event.y, total_levels - max_display_levels))

async def main():
    screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
    pygame.display.set_caption("Sokoban")
    clock = pygame.time.Clock()
    selected_option = -1
    while True:
        mouse_pos = pygame.mouse.get_pos()
        buttons = draw_menu(screen, selected_option, mouse_pos)
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % 3
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % 3
                elif event.key == pygame.K_RETURN:
                    if selected_option == 0:
                        game = Game(level=0)
                        result = await game.run()
                        if result == "quit":
                            pygame.quit()
                            sys.exit()
                        screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
                        pygame.display.set_caption("Sokoban")
                        selected_option = -1
                    elif selected_option == 1:
                        level_index = select_level(screen)
                        if level_index is not None:
                            game = Game(level=level_index)
                            result = await game.run()
                            if result == "quit":
                                pygame.quit()
                                sys.exit()
                        pygame.display.set_caption("Sokoban")
                        selected_option = -1
                    elif selected_option == 2:
                        pygame.quit()
                        sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.collidepoint(mouse_pos):
                        selected_option = i
                        if selected_option == 0:
                            game = Game(level=0)
                            result = await game.run()
                            if result == "quit":
                                pygame.quit()
                                sys.exit()
                            screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
                            pygame.display.set_caption("Sokoban")
                            selected_option = -1
                        elif selected_option == 1:
                            level_index = select_level(screen)
                            if level_index is not None:
                                game = Game(level=level_index)
                                result = await game.run()
                                if result == "quit":
                                    pygame.quit()
                                    sys.exit()
                            screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
                            pygame.display.set_caption("Sokoban")
                            selected_option = -1
                        elif selected_option == 2:
                            pygame.quit()
                            sys.exit()
        await asyncio.sleep(1.0 / FPS)

if __name__ == "__main__":
    asyncio.run(main())