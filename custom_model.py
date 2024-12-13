from copy import deepcopy
import numpy as np
import random

class CUSTOM_AI_MODEL:
    def __init__(self, weights=None, nfeatures=4, mutate=False, noise=0.1):
        self.weights = weights
        self.nfeatures = nfeatures
        self.mutate = mutate
        self.noise = noise

        if self.weights is None:
            # Default weights, updated to include the "holes" feature
            # self.weights = np.array([random.uniform(-1, 0) for feature in range(nfeatures)])
            self.weights = np.array([-0.63725271, -0.29199838, -0.07409222, -0.37546867])
        elif mutate:
            self.weights = weights * (np.array([np.random.normal(1, noise) for _ in range(nfeatures)]))

        self.fit_score = 0.0  # Fitness score
        self.fit_rel = 0.0  # Relative fitness compared to other agents

    def __lt__(self, other):
        return self.fit_score < other.fit_score

    def how_good(self, board):
        npeaks = peaks(board)
        nbumpiness = bumpiness(npeaks)
        nholes = sum(get_holes(npeaks, board))  # Total number of holes across all columns

        # Feature vector includes peaks, bumpiness, filled rows, and holes
        ratings = np.array([
            np.sum(npeaks),  # Total peaks
            nbumpiness,  # Bumpiness
            np.count_nonzero(np.mean(board, axis=1)),  # Partially filled rows
            nholes  # Total number of holes
        ])

        # Score the board state by taking the dot product of features and weights
        return np.dot(ratings, self.weights)

    def get_best_move(self, board, piece):
        best_x = -1000
        max_value = -1000
        best_piece = None

        for i in range(4):  # Loop through all possible rotations
            piece = piece.get_next_rotation()
            for x in range(board.width):  # Loop through all possible placements
                try:
                    y = board.drop_height(piece, x)
                except:
                    continue

                # Create a copy of the board and simulate the piece placement
                board_copy = deepcopy(board.board)
                for pos in piece.body:
                    board_copy[y + pos[1]][x + pos[0]] = True

                # Convert board_copy to a numpy array of 0s and 1s
                board_copy = np.asarray([[1 if cell else 0 for cell in row] for row in board_copy])
                c = self.how_good(board_copy)

                # Update the best move if this placement has a higher score
                if c > max_value:
                    max_value = c
                    best_x = x
                    best_piece = piece

        return best_x, best_piece

# New Function: get_holes
def get_holes(peaks, area):
    """
    Count the number of empty cells (holes) below the peaks in each column.
    """
    holes = []
    for col in range(area.shape[1]):
        start = -peaks[col]  # Topmost filled block in the column
        if start == 0:  # No blocks in the column
            holes.append(0)
        else:
            holes.append(np.count_nonzero(area[int(start):, col] == 0))
    return holes

def peaks(board):
    """
    Calculate the height of the tallest block in each column of the board.
    """
    peaks = np.array([]) 
    nrow, ncol = board.shape[1], board.shape[0]

    for col in range(nrow):
        if 1 in board[:, col]:  
            k = ncol - np.argmax(board[:, col][::-1], axis=0)
            peaks = np.append(peaks, k)
        else:
            peaks = np.append(peaks, 0)
    return peaks

def bumpiness(npeaks):
    """
    Calculate the total bumpiness (differences in adjacent column heights).
    """
    total = 0
    for i in range(len(npeaks) - 1):
        total += np.abs(npeaks[i] - npeaks[i + 1])
    return total