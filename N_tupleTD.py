import copy
import random
import math
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from game2048_env import Game2048Env


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rot90(pattern):
    return [(y, 3 - x) for x, y in pattern]

def rot180(pattern):
    return [(3 - x, 3 - y) for x, y in pattern]

def rot270(pattern):
    return [(3 - y, x) for x, y in pattern]

def flip_horizontal(pattern):
    return [(x, 3 - y) for x, y in pattern]


# Returns the board state and score before adding a new random tile
def step_Ntuple(env, action):
    env_copy = copy.deepcopy(env)
    # Call the corresponding move function directly instead of env.step()
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


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        # Here we store symmetry patterns as groups (one list per original pattern)
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        syms = []
        seen = set()  # use a set of tuple-of-tuples to track uniqueness

        transforms = [lambda x: x, rot90, rot180, rot270]
        for t in transforms:
            rotated = t(pattern)
            for p in [rotated, flip_horizontal(rotated)]:
                p_tuple = tuple(p)  # convert list of tuples to a tuple of tuples
                if p_tuple not in seen:
                    seen.add(p_tuple)
                    syms.append(p)

        return syms

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        # Use numpy array indexing: board[x, y]
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0.0
        # For each pattern group, average over the symmetric transformations
        for weight_dict, sym_group in zip(self.weights, self.symmetry_patterns):
            group_value = 0.0
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                group_value += weight_dict[feature]
            total_value += group_value / len(sym_group)
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        # Update each pattern group by averaging the update over all symmetric features.
        for weight_dict, sym_group in zip(self.weights, self.symmetry_patterns):
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                weight_dict[feature] += (alpha * delta) / len(sym_group)


def td_learning(env, approximator, num_episodes=50000, alpha=0.01, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning with afterstate evaluation (gamma=1).
    
    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        epsilon: Epsilon-greedy exploration rate. (Unused if purely greedy)
    """
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        previous_score = env.score
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # -------------------------------
            # Pure greedy action selection using reward + value
            # -------------------------------
            best_value = -float('inf')
            best_action = None
            for a in legal_moves:
                afterstate_board, score_after, moved = step_Ntuple(env, a)
                if not moved:
                    continue
                reward_estimated = score_after - previous_score
                val_est = reward_estimated + approximator.value(afterstate_board)
                if val_est > best_value:
                    best_value = val_est
                    best_action = a

            action = best_action

            # Step into the afterstate (no random tile yet)
            s_prime, score_after, moved = step_Ntuple(env, action)
            reward = score_after - previous_score
            previous_score = score_after

            # Then perform full environment step (adds random tile, sets next state)
            s_double_prime, _, done, _ = env.step(action)
            max_tile = max(max_tile, np.max(s_double_prime))

            # -------------------------------
            # Compute the value of the afterstate from the next chosen action
            # (if not terminal). This is the "next" afterstate in the paper's pseudocode.
            # -------------------------------
            legal_next_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_next_moves:
                best_next_value = 0
                best_next_reward = 0
            else:
                best_next_value = -float('inf')
                best_next_reward = 0
                for a_prime in legal_next_moves:
                    next_afterstate_board, next_score_after, next_moved = step_Ntuple(env, a_prime)
                    if not next_moved:
                        continue
                    # reward from the next action
                    r_next = next_score_after - previous_score
                    val_next = r_next + approximator.value(next_afterstate_board)
                    if val_next > best_next_value:
                        best_next_value = val_next
                        best_next_reward = r_next

            # s'_next: afterstate board for the next best action
            # We only need its value (no discount).
            # We'll find that by reusing best_next_value minus the immediate reward:
            V_s_prime_next = best_next_value - best_next_reward if best_next_value > -float('inf') else 0

            # -------------------------------
            # TD(0) update on the afterstate value (gamma=1)
            # The paper's formula: V(s') <- V(s') + alpha * (r_next + V(s'_next) - V(s'))
            # where r_next is the next action's reward
            # -------------------------------
            V_s_prime = approximator.value(s_prime)
            td_error = best_next_reward + V_s_prime_next - V_s_prime
            approximator.update(s_prime, td_error, alpha)

            # Update the environment's "state" to reflect the real game state (s'')
            state = s_double_prime

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores


# -------------------------------
# TODO: Define your own n-tuple patterns
# -------------------------------
patterns = [
    # === Original 4-tuples ===
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # Vertical line
    [(1, 0), (1, 1), (1, 2), (1, 3)],  # Horizontal line
    [(2, 0), (3, 0), (2, 1), (3, 1)],  # 2x2 bottom-left square
    [(1, 0), (2, 0), (1, 1), (2, 1)],  # 2x2 center-left square
    [(1, 1), (2, 1), (1, 2), (2, 2)],  # 2x2 center square

    # === New 6-tuples from image ===
    [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)],  # Red box (left 2x3)
    [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)],  # Blue box (center 2x3)
    [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)],  # Green Z-ish
    [(0, 1), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)],  # Purple backward Z
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training (gamma=1, i.e. no discounting)
final_scores = td_learning(env, approximator, num_episodes=30000, alpha=0.1, epsilon=0.001)

# Save the weights to a file for later use
with open("weights.pkl", "wb") as f:
    pickle.dump(approximator.weights, f)

# Apply a moving average to smooth the curve
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_scores = moving_average(final_scores, window_size=50)
plt.figure(figsize=(10, 5))
plt.plot(final_scores, label="Raw Score", alpha=0.3)
plt.plot(range(49, len(final_scores)), smoothed_scores, label="Smoothed (50-avg)", color='red')
plt.title("TD(0) Learning - Final Scores (Afterstate Updates)")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.show()
