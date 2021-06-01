#ifndef MANIFOLD_FITTER_HPP_
#define MANIFOLD_FITTER_HPP_

#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <functional>
#include <new>
#include <random>

using index_t = uint16_t;
using FloatArr = float* __restrict__;
using IndexTArr = index_t* __restrict__;
constexpr size_t kCacheLineSize = 64;


struct SurfaceGeometry {
  /// Computes the canonical difference vector between batches of coordinate
  /// pairs. The difference vector is not necessarily the difference in the
  /// coordinate values, since the surface might have an interesting topology
  /// (e.g. coordinates wrapping across edges).
  /// Arguments: left_coords, right_coords, batch_size, dest
  std::function<void(const FloatArr, const FloatArr,
                     size_t, FloatArr)> CoordDiffs;

  /// Computes the square root of the quadratic form of the metric on a
  /// batch of coordinates and tangent space vectors.
  /// Arguments: coords, vectors, batch_size, dest
  std::function<void(const FloatArr, const FloatArr,
                     size_t, FloatArr)> MetricRootQuadForm;
};


class ManifoldFitter {
 public:
  struct TrainParams {
    float lr;
    float gamma;
    float init_noise;
    int supersample_limit;
    float supersample_mult;
    float loss_epsilon;
    int stagnant_thres;
    int small_grid_limit;
    int small_grid_iters;
    bool print_iters;
    bool print_total_iters;
  };

  void LoadCoords(FloatArr coords, index_t num_coords) {
    coords_ = coords;
    num_coords_ = num_coords;
  }

  void LoadGraph(IndexTArr graph, IndexTArr edge_counts, index_t num_coords,
                 size_t total_edges) {
    graph_ = graph;
    edge_counts_ = edge_counts;
    num_coords_ = num_coords;
    total_edges_ = total_edges;
  }

  void SetGridAspect(float grid_aspect) {
    grid_aspect_ = grid_aspect;
  }

  void SetGeometry(SurfaceGeometry* geometry) {
    surface_geometry_ = geometry;
  }

  void SetTrainParams(TrainParams* train_params) {
    train_params_ = train_params;
  }

  void SetSeed(uint64_t seed) {
    rng_.seed(seed);
  }

  void Fit(FloatArr dest);

  void Interpolate(FloatArr new_coords, index_t num_new_coords,
                   FloatArr dest);

  ManifoldFitter() {
    grid_points_ = nullptr;
  }

  ~ManifoldFitter() {
    free(grid_points_);
  }

 private:
  static void* CacheLineAlloc(size_t size) {
    void* result = aligned_alloc(kCacheLineSize, size);
    if (result == nullptr) {
      throw std::bad_alloc();
    }
    return result;
  }

  struct WorkingData {
    IndexTArr coord_grid_corners;
    FloatArr coord_grid_surpluses;
    IndexTArr graph_centers;
    FloatArr target_distances;
    FloatArr interpolated_points;
    FloatArr interpolated_points_grad;
    FloatArr grid_points_move;
    FloatArr grid_points_copy;
    FloatArr scratchpad;
  };

  void Prepare(WorkingData& data);
  void PrepareIteration(WorkingData& data);
  float RunIteration(WorkingData& data);
  void Supersample(WorkingData& data, float supersample_mult);
  void Cleanup(WorkingData& data);

  /// Variables provided from the outside.
  FloatArr coords_;
  IndexTArr graph_;
  IndexTArr edge_counts_;
  float grid_aspect_;
  SurfaceGeometry* surface_geometry_;
  TrainParams* train_params_;
  index_t num_coords_;
  size_t total_edges_;

  /// Variables created internally.
  std::default_random_engine rng_;
  float left_;
  float right_;
  float bottom_;
  float top_;
  FloatArr grid_points_;
  index_t grid_width_;
  index_t grid_height_;
};

#endif