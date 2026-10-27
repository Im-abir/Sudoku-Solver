
import time


def is_valid(grid, row, col, num):
    if num in grid[row]:
        return False

    if any(grid[r][col] == num for r in range(9)):
        return False

    br = (row // 3) * 3
    bc = (col // 3) * 3

    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if grid[r][c] == num:
                return False

    return True


def validate_grid(grid):
    if len(grid) != 9:
        return False

    for row in grid:
        if len(row) != 9:
            return False
        if any(int(n) != n or n < 0 or n > 9 for n in row):
            return False

    for r in range(9):
        for c in range(9):
            num = int(grid[r][c])

            if num == 0:
                continue

            grid[r][c] = 0
            valid = is_valid(grid, r, c, num)
            grid[r][c] = num

            if not valid:
                return False

    return True


def solve(grid, max_nodes=100000, max_seconds=60):
    if not validate_grid(grid):
        return False

    start = time.monotonic()
    nodes = [0]

    def backtrack():
        nodes[0] += 1

        if nodes[0] > max_nodes:
            return False

        if time.monotonic() - start > max_seconds:
            return False

        best = None
        best_options = None

        for r in range(9):
            for c in range(9):
                if grid[r][c] != 0:
                    continue

                options = [
                    n for n in range(1, 10)
                    if is_valid(grid, r, c, n)
                ]

                if not options:
                    return False

                if best_options is None or len(options) < len(best_options):
                    best = (r, c)
                    best_options = options

        if best is None:
            return True

        r, c = best
        assert best_options is not None

        for num in best_options:
            grid[r][c] = num

            if backtrack():
                return True

            grid[r][c] = 0

        return False

    return backtrack()
