def split_cells(warped_gray, margin=8):
    size = warped_gray.shape[0]
    cell_size = size // 9
    cells = []
    for row in range(9):
        for col in range(9):
            y1 = row * cell_size + margin
            y2 = (row + 1) * cell_size - margin
            x1 = col * cell_size + margin
            x2 = (col + 1) * cell_size - margin
            cells.append(warped_gray[y1:y2, x1:x2])
    return cells