# student_agent.py
import numpy as np
import pickle
import random
import gym
from gym import spaces
import math
import os
import copy
from collections import defaultdict

# === N-Tuple helper functions ===
def rot90(pattern):
    return [(y, 3 - x) for x, y in pattern]

def rot180(pattern):
    return [(3 - x, 3 - y) for x, y in pattern]

def rot270(pattern):
    return [(3 - y, x) for x, y in pattern]

def flip_horizontal(pattern):
    return [(x, 3 - y) for x, y in pattern]

# === Step without adding random tile ===
def step_Ntuple(env, action):
    env_copy = copy.deepcopy(env)
    if action == 0:
        moved = env_copy.move_up()
    elif action == 1:
        moved = env_copy.move_down()
    elif action == 2:
        moved = env_copy.move_left()
    elif action == 3:
        moved = env_copy.move_right()
    else:
        moved = False
    return env_copy.board.copy(), env_copy.score, moved

# === NTupleApproximator ===
class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        syms = []
        seen = set()
        transforms = [lambda x: x, rot90, rot180, rot270]
        for t in transforms:
            rotated = t(pattern)
            for p in [rotated, flip_horizontal(rotated)]:
                p_tuple = tuple(p)
                if p_tuple not in seen:
                    seen.add(p_tuple)
                    syms.append(p)
        return syms

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        total_value = 0.0
        for weight_dict, sym_group in zip(self.weights, self.symmetry_patterns):
            group_value = 0.0
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                group_value += weight_dict[feature]
            total_value += group_value / len(sym_group)
        return total_value

# === 2048 Game Environment (minimal for agent use) ===
class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        return np.pad(new_row, (0, self.size - len(new_row)), mode='constant')

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def simulate_row_move(self, row):
        row = self.compress(row)
        row = self.merge(row)
        return self.compress(row)

    def is_move_legal(self, action):
        temp_board = self.board.copy()
        for i in range(self.size):
            if action == 0:
                col = temp_board[:, i]
                new_col = self.simulate_row_move(col)
                if not np.array_equal(col, new_col):
                    return True
            elif action == 1:
                col = temp_board[:, i][::-1]
                new_col = self.simulate_row_move(col)[::-1]
                if not np.array_equal(temp_board[:, i], new_col):
                    return True
            elif action == 2:
                row = temp_board[i]
                new_row = self.simulate_row_move(row)
                if not np.array_equal(row, new_row):
                    return True
            elif action == 3:
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)[::-1]
                if not np.array_equal(temp_board[i], new_row):
                    return True
        return False

    def move_left(self):
        moved = False
        for i in range(self.size):
            orig = self.board[i].copy()
            row = self.simulate_row_move(orig)
            self.board[i] = row
            if not np.array_equal(orig, row):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            orig = self.board[i].copy()
            row = self.simulate_row_move(orig[::-1])[::-1]
            self.board[i] = row
            if not np.array_equal(orig, row):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for i in range(self.size):
            orig = self.board[:, i].copy()
            col = self.simulate_row_move(orig)
            self.board[:, i] = col
            if not np.array_equal(orig, col):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for i in range(self.size):
            orig = self.board[:, i].copy()
            col = self.simulate_row_move(orig[::-1])[::-1]
            self.board[:, i] = col
            if not np.array_equal(orig, col):
                moved = True
        return moved

# === Main Agent Function ===
def get_action(state, score):
    patterns = [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(2, 0), (3, 0), (2, 1), (3, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)],
        [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)],
        [(0, 1), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)],
    ]

    approximator = NTupleApproximator(board_size=4, patterns=patterns)

    # === Load weights safely ===
    if os.path.exists("weights.pkl"):
        try:
            with open("weights.pkl", "rb") as f:
                raw_weights = pickle.load(f)
                approximator.weights = [defaultdict(float, w) for w in raw_weights]
        except Exception as e:
            print("⚠️ Failed to load weights:", e)

    env = Game2048Env()
    env.board = state.copy()
    env.score = score

    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    best_val = -float("inf")
    best_action = None

    for a in legal_moves:
        afterstate_board, score_after, moved = step_Ntuple(env, a)
        if not moved:
            continue
        reward = score_after - env.score
        val = reward + approximator.value(afterstate_board)
        if val > best_val:
            best_val = val
            best_action = a

    return best_action
