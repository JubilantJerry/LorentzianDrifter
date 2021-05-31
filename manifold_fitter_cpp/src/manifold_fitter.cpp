#include "manifold_fitter.hpp"
#include <cassert>
#include <cmath>
#include <cstring>


void ManifoldFitter::Prepare(WorkingData& data) {
  // Finding the minimum and maximum coordinates.
  float* coords = coords_;
  size_t num_coords = num_coords_;
  assert(num_coords_ > 0);
  float left = coords[0];
  float right = coords[0];
  float bottom = coords[1];
  float top = coords[1];
  for (size_t i = 0; i < num_coords; ++i) {
    float x = coords[2 * i];
    float y = coords[2 * i + 1];
    if (x < left) left = x;
    if (x > right) right = x;
    if (y < bottom) bottom = y;
    if (y > top) top = y;
  }
  left_ = left;
  right_ = right;
  bottom_ = bottom;
  top_ = top;

  // Allocating the grid points and related.
  size_t max_grid_width, max_grid_height;
  float max_grid_size = 1.0f;
  float supersample_mult = train_params_->supersample_mult;
  float supersample_limit = static_cast<float>(
      train_params_->supersample_limit);
  while (max_grid_size < supersample_limit) {
    max_grid_size = floor(max_grid_size * supersample_mult) + 1.0f;
  }
  if (grid_aspect_ <= 1.0f) {
    max_grid_height = static_cast<size_t>(max_grid_size);
    max_grid_width = static_cast<size_t>(max_grid_size * grid_aspect_  + 0.5f);
  } else {
    max_grid_width = static_cast<size_t>(max_grid_size);
    max_grid_height = static_cast<size_t>(max_grid_size / grid_aspect_ + 0.5f);
  }
  free(grid_points_);
  // We allocate four floats per point even though we only need 3D points,
  // so that the SIMD loads will always be 16 byte aligned. The working set
  // is likely cache resident so the extra memory usage is fine.
  size_t grid_points_arr_size =
    (max_grid_width + 1) * (max_grid_height + 1) * 4 * sizeof(float);
  grid_points_ = static_cast<FloatArr>(CacheLineAlloc(grid_points_arr_size));
  data.grid_points_move = static_cast<FloatArr>(
    CacheLineAlloc(grid_points_arr_size));
  data.grid_points_copy = static_cast<FloatArr>(
    CacheLineAlloc(grid_points_arr_size));

  // Allocate the interpolated point and grad arrays.
  size_t interpolated_points_arr_size = num_coords_ * 4 * sizeof(float);
  data.interpolated_points = static_cast<FloatArr>(
      CacheLineAlloc(interpolated_points_arr_size));
  data.interpolated_points_grad = static_cast<FloatArr>(
      CacheLineAlloc(interpolated_points_arr_size));

  // Initializing the bilinear grid.
  grid_width_ = 1;
  grid_height_ = 1;
  float* grid_points = grid_points_;
  std::normal_distribution<float> normal(0.0f, train_params_->init_noise);
  grid_points[0 * (2 * 4) + 0 * 4 + 0] = left + normal(rng_);
  grid_points[0 * (2 * 4) + 0 * 4 + 1] = bottom + normal(rng_);
  grid_points[0 * (2 * 4) + 0 * 4 + 2] = normal(rng_);
  grid_points[0 * (2 * 4) + 1 * 4 + 0] = left + normal(rng_);
  grid_points[0 * (2 * 4) + 1 * 4 + 1] = top + normal(rng_);
  grid_points[0 * (2 * 4) + 1 * 4 + 2] = normal(rng_);
  grid_points[1 * (2 * 4) + 0 * 4 + 0] = right + normal(rng_);
  grid_points[1 * (2 * 4) + 0 * 4 + 1] = bottom + normal(rng_);
  grid_points[1 * (2 * 4) + 0 * 4 + 2] = normal(rng_);
  grid_points[1 * (2 * 4) + 1 * 4 + 0] = right + normal(rng_);
  grid_points[1 * (2 * 4) + 1 * 4 + 1] = top + normal(rng_);
  grid_points[1 * (2 * 4) + 1 * 4 + 2] = normal(rng_);

  // Allocating the grid_corners and related.
  data.coord_grid_corners = static_cast<IndexTArr>(
      CacheLineAlloc(num_coords_ * sizeof(index_t)));
  data.coord_grid_surpluses = static_cast<FloatArr>(
      CacheLineAlloc(num_coords_ * 2 * sizeof(float)));

  // Finding the other vertex for each graph edge.
  size_t total_edges = total_edges_;
  index_t* edge_counts = edge_counts_;
  data.graph_centers = static_cast<IndexTArr>(
    CacheLineAlloc(total_edges * sizeof(index_t)));
  for (size_t i = 0, edge_index = 0; i < num_coords; ++i) {
    for (size_t j = 0; j < edge_counts[i]; ++j, ++edge_index) {
      data.graph_centers[edge_index] = i;
    }
  }

  // Finding the target lengths of graph edges.
  index_t* graph_centers = data.graph_centers;
  index_t* graph = graph_;
  FloatArr coord_centers = static_cast<FloatArr>(
    CacheLineAlloc(total_edges * 2 * sizeof(float)));
  FloatArr coord_neighbors = static_cast<FloatArr>(
    CacheLineAlloc(total_edges * 2 * sizeof(float)));
  FloatArr diff_vecs = static_cast<FloatArr>(
    CacheLineAlloc(total_edges * 2 * sizeof(float)));
  FloatArr edge_centers = static_cast<FloatArr>(
    CacheLineAlloc(total_edges * 2 * sizeof(float)));
  data.target_distances = static_cast<FloatArr>(
    CacheLineAlloc(total_edges * sizeof(float)));
  for (size_t i = 0; i < total_edges; ++i) {
    size_t center = graph_centers[i];
    size_t neighbor = graph[i];
    coord_centers[2 * i] = coords[2 * center];
    coord_centers[2 * i + 1] = coords[2 * center + 1];
    coord_neighbors[2 * i] = coords[2 * neighbor];
    coord_neighbors[2 * i + 1] = coords[2 * neighbor + 1];
  }
  surface_geometry_->CoordDiffs(
      coord_neighbors, coord_centers, total_edges, diff_vecs);
  for (size_t i = 0; i < total_edges; ++i) {
    float diff_vec_x = diff_vecs[2 * i];
    float diff_vec_y = diff_vecs[2 * i + 1];
    size_t center = graph_centers[i];
    edge_centers[2 * i] = coords[2 * center] + diff_vec_x * 0.5f;
    edge_centers[2 * i + 1] = coords[2 * center + 1] + diff_vec_y * 0.5f;
  }
  surface_geometry_->MetricRootQuadForm(
    edge_centers, diff_vecs, total_edges, data.target_distances);
  free(coord_centers);
  free(coord_neighbors);
  free(diff_vecs);
  free(edge_centers);
}


void ManifoldFitter::PrepareIteration(WorkingData& data) {
  // Initializing the grid corners for the coordinates.
  float* coords = coords_;
  index_t* coord_grid_corners = data.coord_grid_corners;
  float* coord_grid_surpluses = data.coord_grid_surpluses;
  float left = left_;
  float right = right_;
  float bottom = bottom_;
  float top = top_;
  float inv_width =  1.0f / (right - left);
  float inv_height = 1.0f / (top - bottom);
  size_t grid_width = grid_width_;
  size_t grid_height = grid_height_;
  size_t num_coords = num_coords_;
  for (size_t i = 0; i < num_coords; ++i) {
      float x = coords[2 * i];
      float y = coords[2 * i + 1];
      float x_frac = (x - left) * inv_width;
      float y_frac = (y - bottom) * inv_height;
      float x_surplus = x_frac * grid_width;
      size_t x_ind = static_cast<size_t>(x_surplus);
      if (x_ind < 0) x_ind = 0;
      if (x_ind > grid_width - 1) x_ind = grid_width - 1;
      x_surplus = x_surplus - static_cast<float>(x_ind);
      float y_surplus = y_frac * grid_height;
      size_t y_ind = static_cast<size_t>(y_surplus);
      if (y_ind < 0) y_ind = 0;
      if (y_ind > grid_height - 1) y_ind = grid_height - 1;
      y_surplus = y_surplus - static_cast<float>(y_ind);
      coord_grid_corners[i] = x_ind * (grid_height + 1) + y_ind;
      coord_grid_surpluses[2 * i] = x_surplus;
      coord_grid_surpluses[2 * i + 1] = y_surplus;
  }

  // Reset the optimizer momentum.
  memset(data.grid_points_move, 0, (grid_width + 1) * (grid_height + 1));
}


float ManifoldFitter::RunIteration(WorkingData& data) {
  // Compute the interpolated points.
  float* grid_points = grid_points_;
  size_t grid_width = grid_width_;
  size_t grid_height = grid_height_;
  index_t* coord_grid_corners = data.coord_grid_corners;
  float* coord_grid_surpluses = data.coord_grid_surpluses;
  float* interpolated_points = data.interpolated_points;
  float* interpolated_points_grad = data.interpolated_points_grad;
  size_t num_coords = num_coords_;
  double loss = 0.0f;

  for (size_t i = 0; i < num_coords; ++i) {
    size_t coord_grid_corner = coord_grid_corners[i];
    float x_surplus = coord_grid_surpluses[2 * i];
    float y_surplus = coord_grid_surpluses[2 * i + 1];
    float x_surplus_opp = 1.0f - x_surplus;
    float y_surplus_opp = 1.0f - y_surplus;
    size_t bot_left_index = 4 * coord_grid_corner;
    size_t top_left_index = 4 * coord_grid_corner + 4;
    size_t bot_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1);
    size_t top_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1) + 4;
    interpolated_points[4 * i] =
      x_surplus * y_surplus * grid_points[top_right_index] +
      x_surplus_opp * y_surplus * grid_points[top_left_index] +
      x_surplus * y_surplus_opp * grid_points[bot_right_index] +
      x_surplus_opp * y_surplus_opp * grid_points[bot_left_index];
    interpolated_points[4 * i + 1] =
      x_surplus * y_surplus * grid_points[top_right_index + 1] +
      x_surplus_opp * y_surplus * grid_points[top_left_index + 1] +
      x_surplus * y_surplus_opp * grid_points[bot_right_index + 1] +
      x_surplus_opp * y_surplus_opp * grid_points[bot_left_index + 1];
    interpolated_points[4 * i + 2] =
      x_surplus * y_surplus * grid_points[top_right_index + 2] +
      x_surplus_opp * y_surplus * grid_points[top_left_index + 2] +
      x_surplus * y_surplus_opp * grid_points[bot_right_index + 2] +
      x_surplus_opp * y_surplus_opp * grid_points[bot_left_index + 2];
    interpolated_points_grad[4 * i] = 0.0f;
    interpolated_points_grad[4 * i + 1] = 0.0f;
    interpolated_points_grad[4 * i + 2] = 0.0f;
  }

  // Compute the interpolated point gradients.
  float* target_distances = data.target_distances;
  size_t total_edges = total_edges_;
  index_t* graph_centers = data.graph_centers;
  index_t* graph = graph_;
  for (size_t i = 0; i < total_edges; ++i) {
    size_t center = graph_centers[i];
    size_t neighbor = graph[i];
    float center_x = interpolated_points[4 * center];
    float center_y = interpolated_points[4 * center + 1];
    float center_z = interpolated_points[4 * center + 2];
    float neighbor_x = interpolated_points[4 * neighbor];
    float neighbor_y = interpolated_points[4 * neighbor + 1];
    float neighbor_z = interpolated_points[4 * neighbor + 2];
    float diff_x = neighbor_x - center_x;
    float diff_y = neighbor_y - center_y;
    float diff_z = neighbor_z - center_z;
    float true_distance_sq =
        diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
    float true_distance_inv = 1.0f / sqrtf(true_distance_sq);
    float target_distance = target_distances[i];
    float diff_term = (true_distance_sq * true_distance_inv - target_distance);
    loss += diff_term * diff_term;
    float grad_mult = (1.0f - target_distance * true_distance_inv);
    interpolated_points_grad[4 * center] -= grad_mult * diff_x;
    interpolated_points_grad[4 * center + 1] -= grad_mult * diff_y;
    interpolated_points_grad[4 * center + 2] -= grad_mult * diff_z;
    interpolated_points_grad[4 * neighbor] += grad_mult * diff_x;
    interpolated_points_grad[4 * neighbor + 1] += grad_mult * diff_y;
    interpolated_points_grad[4 * neighbor + 2] += grad_mult * diff_z;
  }

  // Use the gradient to update the parameters and momentum.
  float* grid_points_move = data.grid_points_move;
  float gamma_mult = 1.0f - train_params_->gamma;
  float grad_mult = train_params_->lr * train_params_->gamma / gamma_mult;
  for (size_t i = 0; i < num_coords; ++i) {
    size_t coord_grid_corner = coord_grid_corners[i];
    float x_surplus = coord_grid_surpluses[2 * i];
    float y_surplus = coord_grid_surpluses[2 * i + 1];
    float x_surplus_opp = 1.0f - x_surplus;
    float y_surplus_opp = 1.0f - y_surplus;
    size_t bot_left_index = 4 * coord_grid_corner;
    size_t top_left_index = 4 * coord_grid_corner + 4;
    size_t bot_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1);
    size_t top_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1) + 4;
    grid_points_move[top_right_index] += grad_mult *
      x_surplus * y_surplus * interpolated_points_grad[4 * i];
    grid_points_move[top_right_index + 1] += grad_mult *
      x_surplus * y_surplus * interpolated_points_grad[4 * i + 1];
    grid_points_move[top_right_index + 2] += grad_mult *
      x_surplus * y_surplus * interpolated_points_grad[4 * i + 2];
    grid_points_move[top_left_index] += grad_mult *
      x_surplus_opp * y_surplus * interpolated_points_grad[4 * i];
    grid_points_move[top_left_index + 1] += grad_mult *
      x_surplus_opp * y_surplus * interpolated_points_grad[4 * i + 1];
    grid_points_move[top_left_index + 2] += grad_mult *
      x_surplus_opp * y_surplus * interpolated_points_grad[4 * i + 2];
    grid_points_move[bot_right_index] += grad_mult *
      x_surplus * y_surplus_opp * interpolated_points_grad[4 * i];
    grid_points_move[bot_right_index + 1] += grad_mult *
      x_surplus * y_surplus_opp * interpolated_points_grad[4 * i + 1];
    grid_points_move[bot_right_index + 2] += grad_mult *
      x_surplus * y_surplus_opp * interpolated_points_grad[4 * i + 2];
    grid_points_move[bot_left_index] += grad_mult *
      x_surplus_opp * y_surplus_opp * interpolated_points_grad[4 * i];
    grid_points_move[bot_left_index + 1] += grad_mult *
      x_surplus_opp * y_surplus_opp * interpolated_points_grad[4 * i + 1];
    grid_points_move[bot_left_index + 2] += grad_mult *
      x_surplus_opp * y_surplus_opp * interpolated_points_grad[4 * i + 2];
  }
  for (size_t i = 0; i < (grid_width + 1) * (grid_height + 1); ++i) {
    float move_x = gamma_mult * grid_points_move[4 * i];
    float move_y = gamma_mult * grid_points_move[4 * i + 1];
    float move_z = gamma_mult * grid_points_move[4 * i + 2];
    grid_points[4 * i] -= move_x;
    grid_points[4 * i + 1] -= move_y;
    grid_points[4 * i + 2] -= move_z;
    grid_points_move[4 * i] = move_x;
    grid_points_move[4 * i + 1] = move_y;
    grid_points_move[4 * i + 2] = move_z;
  }
  loss *= 0.5;

  return loss;
}


void ManifoldFitter::Supersample(WorkingData& data, float supersample_mult) {
  // Copy the grid points to another buffer.
  float* grid_points = grid_points_;
  float* grid_points_copy = data.grid_points_copy;
  size_t grid_width = grid_width_;
  size_t grid_height = grid_height_;
  memcpy(grid_points_copy, grid_points,
         (grid_height + 1) * (grid_width + 1) * 4 * sizeof(float));

  // Compute the new grid dimensions.
  size_t new_grid_width = 0;
  size_t new_grid_height = 0;
  if (grid_aspect_ <= 1.0f) {
    new_grid_height = static_cast<size_t>(grid_height_ * supersample_mult) + 1;
    new_grid_width = static_cast<size_t>(
      new_grid_height * grid_aspect_  + 0.5f);
  } else {
    new_grid_width = static_cast<size_t>(grid_width_ * supersample_mult) + 1;
    new_grid_height = static_cast<size_t>(
      new_grid_width / grid_aspect_  + 0.5f);
  }

  // Compute the new grid points.
  for (size_t i = 0, grid_index = 0; i < new_grid_width + 1; i++) {
    float x_frac = static_cast<float>(i) / new_grid_width;
    for (size_t j = 0; j < new_grid_height + 1; j++, grid_index++) {
      float y_frac = static_cast<float>(j) / new_grid_height;
      float x_surplus = x_frac * grid_width;
      size_t x_ind = static_cast<size_t>(x_surplus);
      if (x_ind < 0) x_ind = 0;
      if (x_ind > grid_width - 1) x_ind = grid_width - 1;
      x_surplus = x_surplus - static_cast<float>(x_ind);
      float y_surplus = y_frac * grid_height;
      size_t y_ind = static_cast<size_t>(y_surplus);
      if (y_ind < 0) y_ind = 0;
      if (y_ind > grid_height - 1) y_ind = grid_height - 1;
      y_surplus = y_surplus - static_cast<float>(y_ind);
      size_t coord_grid_corner = x_ind * (grid_height + 1) + y_ind;
      float x_surplus_opp = 1.0f - x_surplus;
      float y_surplus_opp = 1.0f - y_surplus;
      size_t bot_left_index = 4 * coord_grid_corner;
      size_t top_left_index = 4 * coord_grid_corner + 4;
      size_t bot_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1);
      size_t top_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1) + 4;
      grid_points[4 * grid_index] =
        x_surplus * y_surplus * grid_points_copy[top_right_index] +
        x_surplus_opp * y_surplus * grid_points_copy[top_left_index] +
        x_surplus * y_surplus_opp * grid_points_copy[bot_right_index] +
        x_surplus_opp * y_surplus_opp * grid_points_copy[bot_left_index];
      grid_points[4 * grid_index + 1] =
        x_surplus * y_surplus * grid_points_copy[top_right_index + 1] +
        x_surplus_opp * y_surplus * grid_points_copy[top_left_index + 1] +
        x_surplus * y_surplus_opp * grid_points_copy[bot_right_index + 1] +
        x_surplus_opp * y_surplus_opp * grid_points_copy[bot_left_index + 1];
      grid_points[4 * grid_index + 2] =
        x_surplus * y_surplus * grid_points_copy[top_right_index + 2] +
        x_surplus_opp * y_surplus * grid_points_copy[top_left_index + 2] +
        x_surplus * y_surplus_opp * grid_points_copy[bot_right_index + 2] +
        x_surplus_opp * y_surplus_opp * grid_points_copy[bot_left_index + 2];
    }
  }
  grid_width_ = new_grid_width;
  grid_height_ = new_grid_height;
}


void ManifoldFitter::Cleanup(WorkingData& data) {
  free(data.coord_grid_corners);
  free(data.coord_grid_surpluses);
  free(data.graph_centers);
  free(data.target_distances);
  free(data.interpolated_points);
  free(data.interpolated_points_grad);
  free(data.grid_points_move);
  free(data.grid_points_copy);
}


void ManifoldFitter::Interpolate(FloatArr new_coords, index_t num_new_coords,
                                 FloatArr dest) {
  // Compute the new grid points.
  float left = left_;
  float right = right_;
  float bottom = bottom_;
  float top = top_;
  size_t grid_width = grid_width_;
  size_t grid_height = grid_height_;
  float* grid_points = grid_points_;
  float inv_width =  1.0f / (right - left);
  float inv_height = 1.0f / (top - bottom);
  for (size_t i = 0; i < num_new_coords; i++) {
    float x = new_coords[2 * i];
    float y = new_coords[2 * i + 1];
    float x_frac = (x - left) * inv_width;
    float y_frac = (y - bottom) * inv_height;
    float x_surplus = x_frac * grid_width;
    size_t x_ind = static_cast<size_t>(x_surplus);
    if (x_ind < 0) x_ind = 0;
    if (x_ind > grid_width - 1) x_ind = grid_width - 1;
    x_surplus = x_surplus - static_cast<float>(x_ind);
    float y_surplus = y_frac * grid_height;
    size_t y_ind = static_cast<size_t>(y_surplus);
    if (y_ind < 0) y_ind = 0;
    if (y_ind > grid_height - 1) y_ind = grid_height - 1;
    y_surplus = y_surplus - static_cast<float>(y_ind);
    size_t coord_grid_corner = x_ind * (grid_height + 1) + y_ind;
    float x_surplus_opp = 1.0f - x_surplus;
    float y_surplus_opp = 1.0f - y_surplus;
    size_t bot_left_index = 4 * coord_grid_corner;
    size_t top_left_index = 4 * coord_grid_corner + 4;
    size_t bot_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1);
    size_t top_right_index = 4 * coord_grid_corner + 4 * (grid_height + 1) + 4;
    dest[3 * i] =
      x_surplus * y_surplus * grid_points[top_right_index] +
      x_surplus_opp * y_surplus * grid_points[top_left_index] +
      x_surplus * y_surplus_opp * grid_points[bot_right_index] +
      x_surplus_opp * y_surplus_opp * grid_points[bot_left_index];
    dest[3 * i + 1] =
      x_surplus * y_surplus * grid_points[top_right_index + 1] +
      x_surplus_opp * y_surplus * grid_points[top_left_index + 1] +
      x_surplus * y_surplus_opp * grid_points[bot_right_index + 1] +
      x_surplus_opp * y_surplus_opp * grid_points[bot_left_index + 1];
    dest[3 * i + 2] =
      x_surplus * y_surplus * grid_points[top_right_index + 2] +
      x_surplus_opp * y_surplus * grid_points[top_left_index + 2] +
      x_surplus * y_surplus_opp * grid_points[bot_right_index + 2] +
      x_surplus_opp * y_surplus_opp * grid_points[bot_left_index + 2];
  }
}


void ManifoldFitter::Fit(FloatArr dest) {
  WorkingData data;
  Prepare(data);
  float loss_epsilon = train_params_->loss_epsilon;
  size_t small_grid_limit = train_params_->small_grid_limit;
  size_t small_grid_iters = train_params_->small_grid_iters;
  size_t stagnant_thres = train_params_->stagnant_thres;
  float supersample_mult = train_params_->supersample_mult;
  size_t supersample_limit = train_params_->supersample_limit;
  size_t total_iters = 0;
  bool print_iters = train_params_->print_iters;
  bool print_total_iters = train_params_->print_total_iters;

  while (true) {
    size_t stagnant_iters = 0;
    float prev_loss = 0.0f;
    bool has_prev_loss = false;
    size_t iters = 0;
    PrepareIteration(data);

    while (true) {
      float loss = RunIteration(data);
      if (has_prev_loss) {
        float loss_diff = loss - prev_loss;
        if (loss_diff < 0.0f) loss_diff = -loss_diff;
        if (loss_diff < loss_epsilon) {
          ++stagnant_iters;
        } else {
          stagnant_iters = 0;
        }
      }
      ++iters;
      ++total_iters;
      if (print_iters) {
        printf("%0.3f ", loss);
        fflush(stdout);
      }
      bool exit_due_to_small = (
          grid_width_ <= small_grid_limit &&
          grid_height_ <= small_grid_limit &&
          iters >= small_grid_iters);
      bool exit_due_to_stagnant = (
          (grid_width_ > small_grid_limit ||
           grid_height_ > small_grid_limit) &&
          stagnant_iters >= stagnant_thres);
      if (exit_due_to_small || exit_due_to_stagnant) {
        break;
      }
      has_prev_loss = true;
      prev_loss = loss;
    }

    if (print_iters) printf("\n");
    if (grid_width_ >= supersample_limit ||
        grid_height_ >= supersample_limit) {
      break;
    }
    if (grid_width_ <= small_grid_limit &&
        grid_height_ <= small_grid_limit) {
      Supersample(data, 1.0f);
    } else {
      Supersample(data, supersample_mult);
    }
  }
  if (print_total_iters) printf("Total iterations: %lu\n", total_iters);
  Cleanup(data);
  Interpolate(coords_, num_coords_, dest);
}