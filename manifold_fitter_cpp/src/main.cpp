#include <iostream>
#include <cmath>
#include <vector>
#include "manifold_fitter.hpp"

constexpr float kPiFloat = static_cast<float>(M_PI);


void SphereCoordDiffs(const FloatArr left_coords, const FloatArr right_coords,
                      size_t batch_size, FloatArr dest) {
  for (size_t i = 0; i < batch_size; i++) {
    float diff_phi = left_coords[2 * i] - right_coords[2 * i];
    float diff_theta = left_coords[2 * i + 1] - right_coords[2 * i + 1];
    if (diff_theta > kPiFloat) diff_theta -= 2 * kPiFloat;
    if (diff_theta < -kPiFloat) diff_theta += 2 * kPiFloat;
    dest[2 * i] = diff_phi;
    dest[2 * i + 1] = diff_theta;
  }
}


void SphereMetricRootQuadForm(const FloatArr coords, const FloatArr vectors,
                              size_t batch_size, FloatArr dest) {
  for (size_t i = 0; i < batch_size; i++) {
    float phi = coords[2 * i];
    float delta_phi = vectors[2 * i];
    float scaled_delta_theta = sinf(phi) * vectors[2 * i + 1];
    dest[i] = sqrtf(
        delta_phi * delta_phi + scaled_delta_theta * scaled_delta_theta);
  }
}



void SquareGridYWrap(
    float left, float right, float bottom, float top,
    size_t x_count, size_t y_count, std::vector<float>& coords,
    std::vector<index_t>& graph, std::vector<index_t>& edge_counts) {
  size_t row_count = y_count;
  size_t total_size = (x_count + 1) * row_count;
  coords = std::vector<float>(2 * total_size);
  graph = std::vector<index_t>();
  edge_counts = std::vector<index_t>(total_size);
  float x_diff = (right - left) / x_count;
  float y_diff = (top - bottom) / y_count;
  for (size_t i = 0; i < x_count + 1; i++) {
    for (size_t j = 0; j < y_count; j++) {
      size_t index = i * row_count + j;
      size_t edge_count = 0;
      if (i > 0 and j > 0) {
        graph.push_back(index - row_count - 1);
        edge_count += 1;
      }
      if (i > 0 and j == 0) {
        graph.push_back(index - 1);
        edge_count += 1;
      }
      if (i > 0) {
        graph.push_back(index - row_count);
        edge_count += 1;
      }
      if (i > 0 and j < y_count - 1) {
        graph.push_back(index - row_count + 1);
        edge_count += 1;
      }
      if (i > 0 and j == y_count - 1) {
        graph.push_back(index - 2 * row_count + 1);
        edge_count += 1;
      }
      if (j > 0) {
        graph.push_back(index - 1);
        edge_count += 1;
      }
      if (j == 0) {
        graph.push_back(index + row_count - 1);
        edge_count += 1;
      }
      if (j < y_count - 1) {
        graph.push_back(index + 1);
        edge_count += 1;
      }
      if (j == y_count - 1) {
        graph.push_back(index - row_count + 1);
        edge_count += 1;
      }
      if (i < x_count and j > 0) {
        graph.push_back(index + row_count - 1);
        edge_count += 1;
      }
      if (i < x_count and j == 0) {
        graph.push_back(index + 2 * row_count - 1);
        edge_count += 1;
      }
      if (i < x_count) {
        graph.push_back(index + row_count);
        edge_count += 1;
      }
      if (i < x_count and j < y_count - 1) {
        graph.push_back(index + row_count + 1);
        edge_count += 1;
      }
      if (i < x_count and j == y_count - 1) {
        graph.push_back(index + 1);
        edge_count += 1;
      }
      edge_counts[index] = edge_count;
      coords[2 * index] = left + i * x_diff;
      coords[2 * index + 1] = bottom + j * y_diff;
    }
  }
}


int main() {
  ManifoldFitter fitter;
  std::vector<float> coords;
  std::vector<index_t> graph;
  std::vector<index_t> edge_counts;
  SquareGridYWrap(1e-5f, kPiFloat - 1e-5f, -kPiFloat, kPiFloat, 10, 20,
                  coords, graph, edge_counts);
  fitter.LoadCoords(coords.data(), edge_counts.size());
  fitter.LoadGraph(graph.data(), edge_counts.data(),
                   edge_counts.size(), graph.size());
  fitter.SetGridAspect(1.0f);
  SurfaceGeometry geometry = {
    .CoordDiffs = &SphereCoordDiffs,
    .MetricRootQuadForm = &SphereMetricRootQuadForm};
  fitter.SetGeometry(&geometry);
  ManifoldFitter::TrainParams train_params = {
    .lr = 0.1f, .gamma = 0.2f, .init_noise = 0.1f,
    .supersample_limit = 20, .supersample_mult = 1.5f,
    .loss_epsilon = 1.e-5f, .stagnant_thres = 20,
    .small_grid_limit = 5, .small_grid_iters = 100,
    .print_iters = false, .print_total_iters = true
  };
  fitter.SetTrainParams(&train_params);
  std::vector<float> dest(edge_counts.size() * 3);
  for (size_t i = 0; i < 100; i++) {
    fitter.Fit(dest.data());
  }
  // printf("[");
  // for (size_t i = 0; i < edge_counts.size(); i++) {
  //   printf("[%f, %f, %f]", dest[3 * i], dest[3 * i + 1], dest[3 * i + 2]);
  //   if (i != edge_counts.size() - 1) {
  //     printf(", ");
  //   }
  // }
  // printf("]\n");
}