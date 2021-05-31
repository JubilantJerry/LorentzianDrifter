import numpy as np
import torch
import matplotlib.pyplot as plt


class PolarFlatMetric:
    def metric_tensor(self, center):
        return np.array([[1.0, 0.0], [0.0, center[0] ** 2]])

    def x_wrap(self):
        return None

    def y_wrap(self):
        return 2 * np.pi


class CartesianFlatMetric:
    def metric_tensor(self, center):
        return np.eye(2)

    def x_wrap(self):
        return None

    def y_wrap(self):
        return None


class PolarSphereMetric:
    def metric_tensor(self, center):
        return np.array([[1.0, 0.0], [0.0, np.sin(center[0]) ** 2]])

    def x_wrap(self):
        return None

    def y_wrap(self):
        return 2 * np.pi


class StereographicSphereMetric:
    def metric_tensor(self, center):
        divider = (1 + center[0] ** 2 + center[1] ** 2) ** 2
        return np.eye(2) / divider

    def x_wrap(self):
        return None

    def y_wrap(self):
        return None


class PolarHyperbolicMetric:
    def metric_tensor(self, center):
        return np.array([[1.0, 0.0], [0.0, np.sinh(center[0]) ** 2]])

    def x_wrap(self):
        return None

    def y_wrap(self):
        return 2 * np.pi


class PoincareHyperbolicMetric:
    def metric_tensor(self, center):
        divider = (1 - center[0] ** 2 - center[1] ** 2) ** 2
        return np.eye(2) / divider

    def x_wrap(self):
        return None

    def y_wrap(self):
        return None


class KerrEventHorizonMetric:
    def __init__(self, a):
        self.a = a
        self.r = 1 + np.sqrt(1 - a ** 2)

    def metric_tensor(self, center):
        rho_sq = self.r ** 2 + (self.a * np.cos(center[0])) ** 2
        return np.array([
            [rho_sq, 0.0],
            [0.0, (2 * self.r * np.sin(center[0])) ** 2 / rho_sq]])

    def x_wrap(self):
        return None

    def y_wrap(self):
        return 2 * np.pi


def desired_distance(point_1, point_2, metric):
    adjuster = np.zeros(2)
    x_wrap = metric.x_wrap()
    y_wrap = metric.y_wrap()
    if x_wrap is not None:
        x_diff = point_2[0] - point_1[0]
        if x_diff > x_wrap / 2:
            adjuster[0] = x_wrap
        if x_diff < -x_wrap / 2:
            adjuster[0] = -x_wrap
    if y_wrap is not None:
        y_diff = point_2[1] - point_1[1]
        if y_diff > y_wrap / 2:
            adjuster[1] = y_wrap
        if y_diff < -y_wrap / 2:
            adjuster[1] = -y_wrap
    center = (point_1 + adjuster + point_2) / 2
    diff = (point_2 - point_1 - adjuster)
    return np.sqrt(diff @ (metric.metric_tensor(center) @ diff))


class BilinearGrid:
    def __init__(self, left, right, bottom, top, init_noise):
        self.grid_points = torch.nn.Parameter(torch.zeros(2, 2, 3))
        self.grid_points.data[0, :, 0] = left
        self.grid_points.data[1, :, 0] = right
        self.grid_points.data[:, 0, 1] = bottom
        self.grid_points.data[:, 1, 1] = top
        self.grid_points.data += \
            np.random.normal(0, init_noise, size=(2, 2, 3))
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.divisions = 1

    def interpolate_with_points(self, coords, grid_points):
        x_val = coords[:, 0]
        y_val = coords[:, 1]
        x_frac = (x_val - self.left) / (self.right - self.left)
        y_frac = (y_val - self.bottom) / (self.top - self.bottom)
        x_surplus = x_frac * self.divisions
        y_surplus = y_frac * self.divisions
        x_ind = np.clip(x_surplus.astype(np.int64), 0, self.divisions - 1)
        y_ind = np.clip(y_surplus.astype(np.int64), 0, self.divisions - 1)
        x_surplus -= x_ind
        y_surplus -= y_ind
        x_surplus = torch.tensor(
            x_surplus, dtype=torch.float32).unsqueeze(dim=1)
        y_surplus = torch.tensor(
            y_surplus, dtype=torch.float32).unsqueeze(dim=1)
        lower_point = (
            (1 - x_surplus) * grid_points[x_ind, y_ind] +
            x_surplus * grid_points[x_ind + 1, y_ind])
        upper_point = (
            (1 - x_surplus) * grid_points[x_ind, y_ind + 1] +
            x_surplus * grid_points[x_ind + 1, y_ind + 1])
        return (1 - y_surplus) * lower_point + y_surplus * upper_point

    def interpolate(self, coords):
        return self.interpolate_with_points(coords, self.grid_points)

    def forward(self, coords):
        return self.interpolate(coords)

    def supersample(self, mult):
        new_divisions = int(mult * self.divisions) + 1
        x_vals = np.linspace(self.left, self.right, new_divisions + 1)
        y_vals = np.linspace(self.bottom, self.top, new_divisions + 1)
        coord_grid = np.stack(
            np.meshgrid(y_vals, x_vals)[::-1], axis=2).reshape((-1, 2))
        new_grid_points = self.interpolate(coord_grid).view(
            new_divisions + 1, new_divisions + 1, 3)
        self.grid_points = torch.nn.Parameter(new_grid_points)
        self.divisions = new_divisions


def build_indexers(graph):
    length = sum(len(row) for row in graph)
    center = np.zeros(length, dtype=np.int64)
    neighbors = np.zeros(length, dtype=np.int64)
    curr_pos = 0
    for i, row in enumerate(graph):
        len_row = len(row)
        center[curr_pos:curr_pos + len_row] = i
        neighbors[curr_pos:curr_pos + len_row] = row
        curr_pos += len_row
    return (center, neighbors)


def build_desired_dist_vec(coords, graph, metric):
    length = sum(len(row) for row in graph)
    result = torch.zeros(length)
    curr_pos = 0
    for i, row in enumerate(graph):
        for j in row:
            result[curr_pos] = desired_distance(coords[i], coords[j], metric)
            curr_pos += 1
    return result


def metric_diff_vec(interp, indexers, desired_dist_vec):
    true_dist = torch.norm(
        interp[indexers[1]] - interp[indexers[0]], dim=1)
    diff_vec = (true_dist - desired_dist_vec)
    return diff_vec


def metric_loss(interp, indexers, desired_dist_vec):
    diff_vec = metric_diff_vec(interp, indexers, desired_dist_vec)
    return 0.5 * diff_vec @ diff_vec


def square_grid(left, right, bottom, top, x_count, y_count):
    row_count = y_count + 1
    total_size = (x_count + 1) * row_count
    coords = np.zeros((total_size, 2))
    graph = []
    x_diff = (right - left) / x_count
    y_diff = (top - bottom) / y_count
    for i in range(x_count + 1):
        for j in range(y_count + 1):
            index = i * row_count + j
            neighbors = []
            if i > 0 and j > 0:
                neighbors.append(index - row_count - 1)
            if i > 0:
                neighbors.append(index - row_count)
            if i > 0 and j < y_count:
                neighbors.append(index - row_count + 1)
            if j > 0:
                neighbors.append(index - 1)
            if j < y_count:
                neighbors.append(index + 1)
            if i < x_count and j > 0:
                neighbors.append(index + row_count - 1)
            if i < x_count:
                neighbors.append(index + row_count)
            if i < x_count and j < y_count:
                neighbors.append(index + row_count + 1)
            coords[index, 0] = left + i * x_diff
            coords[index, 1] = bottom + j * y_diff
            graph.append(neighbors)
    return coords, graph


def square_grid_ywrap(left, right, bottom, top, x_count, y_count):
    row_count = y_count
    total_size = (x_count + 1) * row_count
    coords = np.zeros((total_size, 2))
    graph = []
    x_diff = (right - left) / x_count
    y_diff = (top - bottom) / y_count
    for i in range(x_count + 1):
        for j in range(y_count):
            index = i * row_count + j
            neighbors = []
            if i > 0 and j > 0:
                neighbors.append(index - row_count - 1)
            if i > 0 and j == 0:
                neighbors.append(index - 1)
            if i > 0:
                neighbors.append(index - row_count)
            if i > 0 and j < y_count - 1:
                neighbors.append(index - row_count + 1)
            if i > 0 and j == y_count - 1:
                neighbors.append(index - 2 * row_count + 1)
            if j > 0:
                neighbors.append(index - 1)
            if j == 0:
                neighbors.append(index + row_count - 1)
            if j < y_count - 1:
                neighbors.append(index + 1)
            if j == y_count - 1:
                neighbors.append(index - row_count + 1)
            if i < x_count and j > 0:
                neighbors.append(index + row_count - 1)
            if i < x_count and j == 0:
                neighbors.append(index + 2 * row_count - 1)
            if i < x_count:
                neighbors.append(index + row_count)
            if i < x_count and j < y_count - 1:
                neighbors.append(index + row_count + 1)
            if i < x_count and j == y_count - 1:
                neighbors.append(index + 1)
            coords[index, 0] = left + i * x_diff
            coords[index, 1] = bottom + j * y_diff
            graph.append(neighbors)
    return coords, graph


def filter_coords(coords, graph, criterion):
    new_coords = []
    index_mapper = {}
    for i in range(coords.shape[0]):
        if criterion(coords[i]):
            index_mapper[i] = len(new_coords)
            new_coords.append(i)
    new_coords = coords[new_coords]
    new_graph = []
    for i, row in enumerate(graph):
        if i not in index_mapper:
            continue
        new_row = []
        for neighbor in row:
            if neighbor not in index_mapper:
                continue
            new_row.append(index_mapper[neighbor])
        new_graph.append(new_row)
    return new_coords, new_graph


def fit(coords, graph, metric, train_params):
    left = np.min(coords[:, 0])
    right = np.max(coords[:, 0])
    bottom = np.min(coords[:, 1])
    top = np.max(coords[:, 1])
    grid = BilinearGrid(left, right, bottom, top, train_params['init_noise'])
    indexers = build_indexers(graph)
    desired_dist_vec = build_desired_dist_vec(coords, graph, metric)

    print_iters = 'print_iters' in train_params and train_params['print_iters']
    lr = train_params['lr']
    gamma = train_params['gamma']
    loss_epsilon = train_params['loss_epsilon']
    stagnant_thres = train_params['stagnant_thres']
    small_grid_limit = train_params['small_grid_limit']
    small_grid_iters = train_params['small_grid_iters']

    total_iters = 0
    s_iter = -1
    while True:
        s_iter += 1
        grid_size = grid.grid_points.shape[0] - 1
        iters = 0
        stagnant_iters = 0
        prev_loss_val = None
        move = None

        while True:
            interp = grid.interpolate(coords)
            loss = metric_loss(interp, indexers, desired_dist_vec)
            loss_val = loss.item()
            if grid.grid_points.grad is not None:
                grid.grid_points.grad.zero_()
            loss.backward()
            if move is None:
                move = gamma * grid.grid_points.grad
                grid.grid_points.data -= lr * move
            else:
                move = move * (1 - gamma)
                move = move + gamma * grid.grid_points.grad
                grid.grid_points.data -= lr * move

            if print_iters:
                print('%0.3f ' % loss_val, end='', flush=True)
            iters += 1
            total_iters += 1
            if prev_loss_val is not None and \
                    abs(loss_val - prev_loss_val) < loss_epsilon:
                stagnant_iters += 1
            else:
                stagnant_iters = 0
            exit_due_to_small = (grid_size <= small_grid_limit and
                                 iters >= small_grid_iters)
            exit_due_to_stagnant = (grid_size > small_grid_limit and
                                    stagnant_iters >= stagnant_thres)
            if (exit_due_to_small or exit_due_to_stagnant):
                break
            prev_loss_val = loss_val
        if print_iters:
            print('')
        if grid_size >= train_params['supersample_limit']:
            break
        if grid_size <= small_grid_limit:
            grid.supersample(1.0)
        else:
            grid.supersample(train_params['supersample_mult'])
    if 'print_total_iters' in train_params and \
            train_params['print_total_iters']:
        print('Total iterations: {}'.format(total_iters))
    return grid.interpolate(coords).data


def plot_3d(points):
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    box_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*points.T)
    x_mid = (x_max + x_min) / 2
    ax.set_xlim(x_mid - box_size / 2, x_mid + box_size / 2)
    y_mid = (y_max + y_min) / 2
    ax.set_ylim(y_mid - box_size / 2, y_mid + box_size / 2)
    z_mid = (z_max + z_min) / 2
    ax.set_zlim(z_mid - box_size / 2, z_mid + box_size / 2)
    ax.set_box_aspect((1, 1, 1))
    plt.show()
