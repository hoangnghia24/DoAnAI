import time
import random
from typing import List, Tuple, Optional, Set, FrozenSet

def save_ga_solution(level_idx, path,elapsed_time):
    if path is None:
        return
    with open("solutions.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Level {level_idx} ---\n")
        f.write("--- Nhóm 3 ---\n")
        f.write(f"Thời gian: {elapsed_time:.10f} giây\n")
        f.write(f"Số bước Genetic: {len(path)}\n")
        f.write(f"Path Genetic: {path}\n\n")
    print(f"GA: Đã lưu lời giải cho Level {level_idx} vào solutions.txt")

def solve_with_genetic_algorithm(level_data: List[List[str]], level_idx: int,
                                 population_size=100, num_generations=50,
                                 chromosome_length=150, mutation_rate=0.05):
    def trim_solution(chromosome: List[int]) -> List[int]:
        player_pos, boxes_pos = initial_player_pos, initial_boxes

        if goals.issubset(boxes_pos):
            return []

        for i, action in enumerate(chromosome):
            dx, dy = directions[action]
            next_player_pos = (player_pos[0] + dx, player_pos[1] + dy)
            if next_player_pos in walls: continue

            if next_player_pos in boxes_pos:
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                if next_box_pos in walls or next_box_pos in boxes_pos: continue
                box_list = list(boxes_pos)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                boxes_pos = frozenset(sorted(box_list))

            player_pos = next_player_pos

            if goals.issubset(boxes_pos):
                return chromosome[:i+1]

        return chromosome

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
    initial_boxes = get_boxes(level_data)
    goals = get_goals(level_data)
    walls = get_walls(level_data)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def calculate_fitness(chromosome: List[int]) -> float:
        player_pos, boxes_pos = initial_player_pos, initial_boxes

        for action in chromosome:
            dx, dy = directions[action]
            next_player_pos = (player_pos[0] + dx, player_pos[1] + dy)
            if next_player_pos in walls: continue

            if next_player_pos in boxes_pos:
                next_box_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                if next_box_pos in walls or next_box_pos in boxes_pos: continue
                box_list = list(boxes_pos)
                box_list.remove(next_player_pos)
                box_list.append(next_box_pos)
                boxes_pos = frozenset(sorted(box_list))

            player_pos = next_player_pos

        if goals.issubset(boxes_pos):
            return 10000.0

        boxes_on_goal = len(boxes_pos.intersection(goals))

        total_dist = 0
        for box in boxes_pos:
            total_dist += min(abs(box[0] - g[0]) + abs(box[1] - g[1]) for g in goals)

        fitness = (boxes_on_goal * 100) + (1.0 / (1.0 + total_dist))
        return fitness

    def selection(population_with_fitness: List[Tuple[List[int], float]]) -> List[int]:
        tournament_size = 5
        best_in_tournament = max(random.sample(population_with_fitness, tournament_size), key=lambda item: item[1])
        return best_in_tournament[0]

    def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutation(chromosome: List[int]) -> List[int]:
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = random.randint(0, 3)
        return chromosome

    print(f"GA: Bắt đầu giải Level {level_idx}...")
    start_time = time.time()

    population = [[random.randint(0, 3) for _ in range(chromosome_length)] for _ in range(population_size)]

    for generation in range(num_generations):
        population_with_fitness = [(chromo, calculate_fitness(chromo)) for chromo in population]
        population_with_fitness.sort(key=lambda item: item[1], reverse=True)

        best_fitness = population_with_fitness[0][1]

        if best_fitness >= 10000.0:
            best_solution = population_with_fitness[0][0]
            trimmed_solution = trim_solution(best_solution)
            elapsed_time = time.time() - start_time
            print(f"GA: Tìm thấy lời giải sau {elapsed_time:.2f} giây.")
            save_ga_solution(level_idx, trimmed_solution,elapsed_time)
            return trimmed_solution

        new_population = [population_with_fitness[0][0]]

        while len(new_population) < population_size:
            parent1 = selection(population_with_fitness)
            parent2 = selection(population_with_fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1))
            if len(new_population) < population_size:
                new_population.append(mutation(child2))

        population = new_population

    print(f"GA: Không tìm thấy lời giải cho Level {level_idx} trong giới hạn.")
    return None

