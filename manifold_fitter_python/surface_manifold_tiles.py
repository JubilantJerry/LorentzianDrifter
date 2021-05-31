import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import torch
import ipdb

device = 'cpu'


def metric_mat_to_transform(matrix):
    evl, evc = np.linalg.eigh(matrix)
    return evc @ (np.sqrt(np.abs(evl)) * evc).T


class Cell:
    def __init__(self, index, center, size):
        self.index = index
        self.center = center
        self.size = size
        self.neighbors = []

    def add_neighbor(self, other):
        self.neighbors.append(other)

    def __repr__(self):
        return 'Cell(index={}, center={}, size={}, neighbors={})'.format(
            self.index, self.center, self.size,
            [n.index for n in self.neighbors])


def flat_grid(n, m, v1_range, v2_range):
    interval_v1 = (v1_range[1] - v1_range[0]) / n
    interval_v2 = (v2_range[1] - v2_range[0]) / m
    cell_size = np.array([interval_v1, interval_v2])
    cells = []
    for i in range(n):
        for j in range(m):
            index = i * m + j
            v1_center = (i + 0.5) * interval_v1 + v1_range[0]
            v2_center = (j + 0.5) * interval_v2 + v2_range[0]
            cells.append(Cell(
                index, np.array([v1_center, v2_center]), cell_size))
    for i in range(n):
        for j in range(m):
            index = i * m + j
            if i != 0:
                cells[index].add_neighbor(cells[index - m])
            if i != n - 1:
                cells[index].add_neighbor(cells[index + m])
            if j != 0:
                cells[index].add_neighbor(cells[index - 1])
            if j != m - 1:
                cells[index].add_neighbor(cells[index + 1])
    return cells


def flat_metric(center):
    return np.eye(2)


def sphere_metric(center):
    return np.array([[1.0, 0.0], [0.0, np.sin(center[0]) ** 2]])


def rotations_from_params(rotation_params):
    tril_ind = np.tril_indices(3, -1)
    skew = torch.zeros(rotation_params.shape[0], 3, 3, device=device)
    skew[:, tril_ind[0], tril_ind[1]] = rotation_params
    skew = skew - torch.transpose(skew, 1, 2)
    eye = torch.eye(3)
    rotations = torch.matmul(eye - skew, torch.inverse(eye + skew))
    return rotations


def grid_optimization(grid_cells, metric, dist_term, pos_term, train_params):
    num_cells = len(grid_cells)
    rotation_params = torch.randn(
        num_cells, 3, device=device, requires_grad=True)
    positions = torch.randn(
        num_cells, 3, device=device, requires_grad=True)
    transforms = [
        metric_mat_to_transform(metric(cell.center)) for cell in grid_cells]
    steps, lr, gamma = train_params

    optim = torch.optim.SGD(
        [rotation_params, positions],
        lr=lr * gamma, momentum=(1 - gamma))

    for i in range(steps):
        rotations = rotations_from_params(rotation_params)
        objective = 0
        for cell in grid_cells:
            cell_rot = rotations[cell.index]
            for neighbor in cell.neighbors:
                dist = torch.norm(
                    positions[neighbor.index] - positions[cell.index])
                coord_move = neighbor.center - cell.center
                local_move = transforms[cell.index] @ coord_move
                dist_target = np.linalg.norm(local_move)
                objective = objective + \
                    dist_term * (dist - dist_target) ** 2
                neighbor_rot = rotations[neighbor.index]
                global_move = 0.5 * (cell_rot + neighbor_rot)[:, :2].matmul(
                    torch.tensor(local_move).float())
                position_diff = \
                    positions[cell.index] + global_move - \
                    positions[neighbor.index]
                objective = objective + \
                    pos_term * torch.abs(position_diff).sum()

                rotation_diff = \
                    rotations[cell.index] - rotations[neighbor.index]
                objective = objective + torch.matmul(
                    rotation_diff.view(1, 9),
                    rotation_diff.view(9, 1))
        objective = objective / num_cells
        print(objective.data.item())
        optim.zero_grad()
        objective.backward()
        optim.step()

    rotations = rotations_from_params(rotation_params)
    return positions.data.numpy(), rotations.data.numpy()


def plot_3d(grid_cells, metric, positions, rotations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    for cell in grid_cells:
        center = positions[cell.index]
        transform = metric_mat_to_transform(metric(cell.center))
        coord_points = cell.size / 2 * np.array([
            [1, 1],
            [1, -1],
            [-1, -1],
            [-1, 1]
        ])
        local_points = coord_points @ transform.T
        global_points = center + local_points @ rotations[cell.index][:, :2].T
        patch = art3d.Poly3DCollection([global_points])
        patch.set_alpha(0.75)
        ax.add_collection3d(patch)
    plt.show()
