def conv_so3_constr(rel_rotations, i, j, constr):
    submat = rel_rotations[3 * i: 3 * i + 3, 3 * j: 3 * j + 3]
    semidef_mat = cp.bmat([
        [1 - submat[0, 0] - submat[1, 1] + submat[2, 2],
         submat[0, 2] + submat[2, 0],
         submat[0, 1] - submat[1, 0],
         submat[1, 2] + submat[2, 1]],
        [submat[0, 2] + submat[2, 0],
         1 + submat[0, 0] - submat[1, 1] - submat[2, 2],
         submat[1, 2] - submat[2, 1],
         submat[0, 1] + submat[1, 0]],
        [submat[0, 1] - submat[1, 0],
         submat[1, 2] - submat[2, 1],
         1 + submat[0, 0] + submat[1, 1] + submat[2, 2],
         submat[2, 0] - submat[0, 2]],
        [submat[1, 2] + submat[2, 1],
         submat[0, 1] + submat[1, 0],
         submat[2, 0] - submat[0, 2],
         1 - submat[0, 0] + submat[1, 1] - submat[2, 2]]])
    constr.append(semidef_mat >> 0)


def grid_optimization(grid_cells, metric, pos_term):
    num_cells = len(grid_cells)
    positions = cp.Variable((num_cells, 3))
    rel_rotations = cp.Variable((3 * num_cells, 3 * num_cells), PSD=True)
    objective = 0
    constr = []
    for cell in grid_cells:
        i = cell.index
        diag_submat = rel_rotations[3 * i: 3 * i + 3, 3 * i: 3 * i + 3]
        constr.append(diag_submat == np.eye(3))
        for j in range(num_cells):
            if j > i:
                conv_so3_constr(rel_rotations, i, j, constr)
        cell_rot = rel_rotations[0: 3, 3 * i: 3 * i + 2]
        for neighbor in cell.neighbors:
            j = neighbor.index
            transform = metric_mat_to_transform(metric(cell.center))
            coord_move = neighbor.center - cell.center
            local_move = transform @ coord_move / 2
            global_move = cell_rot @ local_move
            neighbor_rot = rel_rotations[0: 3, 3 * j: 3 * j + 2]
            transform = metric_mat_to_transform(metric(neighbor.center))
            local_move = transform @ coord_move / 2
            global_move = global_move + neighbor_rot @ local_move
            objective = objective + pos_term * cp.norm2(
                (positions[j] - positions[i]) - global_move) ** 2

            submat = rel_rotations[3 * i: 3 * i + 3, 3 * j: 3 * j + 3]
            objective = objective - cp.trace(submat)
    problem = cp.Problem(cp.Minimize(objective), constr)
    problem.solve(verbose=True, eps=1e-8)
    rotations = rel_rotations.value[:3]
    rotations = rotations.reshape((3, num_cells, 3)).transpose((1, 0, 2))
    for i in range(num_cells):
        u, _, vh = np.linalg.svd(rotations[i])
        rotations[i] = u @ vh
    return positions.value, rotations

