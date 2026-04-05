#!/usr/bin/env python3
"""


"""

from collections import deque
import random
import sys

from abc import ABC, abstractmethod

import heapq

DIRECTIONS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

import math

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def connected(self, x, y):
        return self.find(x) == self.find(y)

def find_bridges(board, color, size):
    """
    Find all virtual bridge pairs for a given color.
    Two stones are bridged if they share exactly 2 empty common neighbors.
    complexity: O(size^4) worst case, but typically much less due to few groups of stones. 
    Mostly O(size^2 * k)
    memory: O(size^2) for union-find and board representation.
    """
    uf = UnionFind()

    # first union all directly adjacent same-color stones
    for (r, c), col in board.items():
        if col != color:
            continue
        for nr, nc in neighbors(r, c, size):
            if board.get((nr, nc)) == color:
                uf.union((r, c), (nr, nc))

    # then detect bridges
    stones = [(r, c) for (r, c), col in board.items() if col == color]
    for i, a in enumerate(stones):
        for b in stones[i+1:]:
            if uf.connected(a, b):
                continue  # already connected, no need to bridge
            # find common empty neighbors
            neighbors_a = set(neighbors(a[0], a[1], size))
            neighbors_b = set(neighbors(b[0], b[1], size))
            common = [
                cell for cell in neighbors_a & neighbors_b
                if board.get(cell) is None
            ]
            if len(common) == 2:
                uf.union(a, b)  # virtually connected

    return uf

def softmax(scores: list[float], temperature: float = 0.1) -> list[float]:
    """
    covert score to probability, higher temperature = more random, lower = more greedy.
    To address normalization issue
    """
    scores = [s / temperature for s in scores]
    max_s = max(scores)
    exp_scores = [math.exp(s - max_s) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def neighbors(row, col, size):
    return [
        (row + dr, col + dc)
        for dr, dc in DIRECTIONS
        if 0 <= row + dr < size and 0 <= col + dc < size
    ]
def opponent(color):
    return 'B' if color == 'R' else 'R'


def bfs_01(board, color, size, from_end=False, use_bridges=True):
    """
    Special BFS treating 0 cost cells by queueing on the front and 1 cost cells on the back.
    If use_bridges is True, treat virtually bridged cells as 0 cost.
    Returns distance from the starting edge to every cell, considering current board and optional bridges.
    If from_end is True, calculates distance from the opposite edge (used for combined path length)
    Complexity: O(size^2) since each cell is visited at most once with its best cost.
    Memory: O(size^2) for distance array and queue in worst case.
    """
    uf = find_bridges(board, color, size) if use_bridges else None

    dist = [[float('inf')] * size for _ in range(size)]
    queue = deque()

    if color == 'R':
        start_cells = [(size-1, c) for c in range(size)] if from_end else [(0, c) for c in range(size)]
    else:
        start_cells = [(r, size-1) for r in range(size)] if from_end else [(r, 0) for r in range(size)]

    for r, c in start_cells:
        if board.get((r, c)) == opponent(color):
            continue
        cost = 0 if board.get((r, c)) == color else 1
        if cost < dist[r][c]:
            dist[r][c] = cost
            queue.append((r, c, cost))

    while queue:
        r, c, d = queue.popleft()
        if d > dist[r][c]:
            continue
        for nr, nc in neighbors(r, c, size):
            if board.get((nr, nc)) == opponent(color):
                continue
            if board.get((nr, nc)) == color:
                cost = 0  # own stone always free, bridge doesn't change this
            elif uf and uf.connected((r, c), (nr, nc)):
                cost = 0  # bridge cells treated as free
            else:
                cost = 1
            new_dist = d + cost
            if new_dist < dist[nr][nc]:
                dist[nr][nc] = new_dist
                if cost == 0:
                    queue.appendleft((nr, nc, new_dist))
                else:
                    queue.append((nr, nc, new_dist))

    return dist



class Heuristic(ABC):
    def prepare(self, board: dict, color: str, size: int):
        """
        Called once per turn before any score() calls.
        Run expensive multiple-cell algorithm here
        """
        pass

    @abstractmethod
    def score(self, board: dict, cell: tuple, color: str, size: int) -> float:
        """Score a single cell. prepare() is always called first."""
        pass

class HeuristicEvaluator:
    def __init__(self, temperature: float = 0.1):
        self.heuristics: list[tuple[Heuristic, float]] = []
        self.temperature = temperature
        self.prepared = False

    def register(self, heuristic: Heuristic, weight: float = 1.0):
        self.heuristics.append((heuristic, weight))
        return self
    
    def score_cell(self, board, cell, color, size):
        if not self.prepared:
            for h, _ in self.heuristics:
                h.prepare(board, color, size)
            self.prepared = True
        return sum(
            weight * h.score(board, cell, color, size)
            for h, weight in self.heuristics
        )

    def best_move(self, board, color, size):
        """
        Determine the best move based on the registered heuristics.
        Apply softmax to combine heuristic scores into probabilities, then sample a move.
        complexity: O(size^2 * h) where h is the number of heuristics, since we score each empty cell with each heuristic.
        memory: O(size^2) for empty cell list and combined scores.
        """
        empty_cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) not in board
        ]

        if not empty_cells:
            return None

        # Prepare heuristics results, cost according to heuristics
        for h, _ in self.heuristics:
            h.prepare(board, color, size)
        self.prepared = True

        combined = [0.0] * len(empty_cells)
        for h, weight in self.heuristics:
            raw = [h.score(board, cell, color, size) for cell in empty_cells]
            normalized = softmax(raw, self.temperature)
            combined = [c + weight * n for c, n in zip(combined, normalized)]

        total = sum(combined)
        probs = [c / total for c in combined]
        return random.choices(empty_cells, weights=probs, k=1)[0]
    
class MyShortestPath(Heuristic):
    def __init__(self, use_bridges=True):
        self.use_bridges = use_bridges

    def prepare(self, board, color, size):
        dist_start = bfs_01(board, color, size, use_bridges=self.use_bridges)
        dist_end   = bfs_01(board, color, size, from_end=True, use_bridges=self.use_bridges)
        self._dist = [
            [dist_start[r][c] + dist_end[r][c] for c in range(size)]
            for r in range(size)
        ]

    def score(self, board, cell, color, size):
        r, c = cell
        return -self._dist[r][c]


class OppShortestPath(Heuristic):
    def __init__(self, use_bridges=True):
        self.use_bridges = use_bridges

    def prepare(self, board, color, size):
        opp = opponent(color)
        dist_start = bfs_01(board, opp, size, use_bridges=self.use_bridges)
        dist_end   = bfs_01(board, opp, size, from_end=True, use_bridges=self.use_bridges)
        self._dist = [
            [dist_start[r][c] + dist_end[r][c] for c in range(size)]
            for r in range(size)
        ]

    def score(self, board, cell, color, size):
        r, c = cell
        return -self._dist[r][c]
    
class CenterBias(Heuristic):
    """Prefer cells closer to the center of the board. 
    complexity: O(size^2) for precomputation
    memory O(size^2) for distance array.
    """
    def prepare(self, board, color, size):
        mid = (size - 1) / 2
        self._dist = [
            [abs(r - mid) + abs(c - mid) for c in range(size)]
            for r in range(size)
        ]

    def score(self, board, cell, color, size):
        r, c = cell
        return -self._dist[r][c]

def parse_board(line):
    """
    Parse the board state from one line.

    Args:
        line: Input line in format "SIZE COLOR MOVES"

    Returns:
        Tuple of (size, my_color, board_dict)
        where board_dict is {(row, col): 'R' or 'B'}
    """
    parts = line.strip().split(maxsplit=2)

    size = int(parts[0])
    my_color = parts[1]  # "RED" or "BLUE"

    # Parse existing moves
    board = {}
    if len(parts) == 3 and parts[2]:
        moves_str = parts[2]
        for move in moves_str.split(','):
            row, col, color = move.split(':')
            board[(int(row), int(col))] = color

    return size, my_color, board

def choose_move(size, my_color, board, evaluator):
    color = 'R' if my_color == 'RED' else 'B'
    return evaluator.best_move(board, color, size)

def should_swap(board, my_color, size, evaluator):
    color = 'R' if my_color == 'RED' else 'B'
    if color != 'B' or len(board) != 1:
        return False

    opp_cell = next(iter(board))
    r, c = opp_cell
    mirrored = (c, r)

    # score swap first
    swapped_board = {}

    swap_score = evaluator.score_cell(swapped_board, mirrored, color, size)

    # score real board second — best_move calls prepare internally
    best_cell = evaluator.best_move(board, color, size)
    best_score = evaluator.score_cell(board, best_cell, color, size)

    return swap_score > best_score

def main():
    evaluator = (
        HeuristicEvaluator(temperature=0.1)
        .register(MyShortestPath(),  weight=1.0) # 2 pass of BFS
        .register(OppShortestPath(), weight=1.0) # 2 pass of BFS
        .register(CenterBias(),      weight=2.0)
    )
    while True:
        try:
            line = input()
            size, my_color, board = parse_board(line)

            if should_swap(board, my_color, size, evaluator):
                print("swap")
            else:
                row, col = choose_move(size, my_color, board, evaluator)
                print(f"{row} {col}")

            sys.stdout.flush()

        except EOFError:
            break


if __name__ == "__main__":
    main()