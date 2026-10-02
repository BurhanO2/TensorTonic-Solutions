import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    board = np.asarray(board)
    size = board.shape[0]
    colour = board[row, col]
    stack = [(row, col)]
    group = set()
    liberties = set()

    while stack:
        point = stack.pop()
        if point in group:
            continue

        group.add(point)
        r, c = point
        for nr, nc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if board[nr, nc] == 0:
                liberties.add((nr, nc))
            elif board[nr, nc] == colour and (nr, nc) not in group:
                stack.append((nr, nc))

    return sorted(group), sorted(liberties)