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
  using CoordDiffsFn =
  	  std::function<void(const FloatArr, const FloatArr, size_t, FloatArr)>;

  using MetricRootQuadFormFn =
  	  std::function<void(const FloatArr, const FloatArr, size_t, FloatArr)>;

  /// Computes the canonical difference vector between batches of coordinate
  /// pairs. The difference vector is not necessarily the difference in the
  /// coordinate values, since the surface might have an interesting topology
  /// (e.g. coordinates wrapping across edges).
  /// Arguments: leftCoords, rightCoords, batchSize, dest
  CoordDiffsFn CoordDiffs;

  /// Computes the square root of the quadratic form of the metric on a
  /// batch of coordinates and tangent space vectors.
  /// Arguments: coords, vectors, batchSize, dest
  MetricRootQuadFormFn MetricRootQuadForm;
};


class ManifoldFitter {
 public:
  struct TrainParams {
    float lr;
    float gamma;
    float initNoise;
    int supersampleLimit;
    float supersampleMult;
    float lossEpsilon;
    int stagnantThres;
    int smallGridLimit;
    int smallGridIters;
    bool printIters;
    bool printTotalIters;
  };

  void LoadCoords(FloatArr coords, index_t numCoords) {
    coords_ = coords;
    numCoords_ = numCoords;
  }

  void LoadGraph(IndexTArr graph, IndexTArr edgeCounts, index_t numCoords,
                 size_t totalEdges) {
    graph_ = graph;
    edgeCounts_ = edgeCounts;
    numCoords_ = numCoords;
    totalEdges_ = totalEdges;
  }

  void SetGridAspect(float gridAspect) {
    gridAspect_ = gridAspect;
  }

  void SetGeometry(SurfaceGeometry* geometry) {
    surfaceGeometry_ = geometry;
  }

  void SetTrainParams(const TrainParams& trainParams) {
    trainParams_ = trainParams;
  }

  void SetSeed(uint64_t seed) {
    rng_.seed(seed);
  }

  void Fit(FloatArr dest);

  void Interpolate(FloatArr newCoords, index_t numNewCoords,
                   FloatArr dest);

  ManifoldFitter() {
    gridPoints_ = nullptr;
  }

  ~ManifoldFitter() {
    free(gridPoints_);
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
    IndexTArr coordGridCorners;
    FloatArr coordGridSurpluses;
    IndexTArr graphCenters;
    FloatArr targetDistances;
    FloatArr interpolatedPoints;
    FloatArr interpolatedPointsGrad;
    FloatArr gridPointsMove;
    FloatArr gridPointsCopy;
    FloatArr scratchpad;
  };

  void Prepare(WorkingData& data);
  void PrepareIteration(WorkingData& data);
  float RunIteration(WorkingData& data);
  void Supersample(WorkingData& data, float supersampleMult);
  void Cleanup(WorkingData& data);

  /// Variables provided from the outside.
  FloatArr coords_;
  IndexTArr graph_;
  IndexTArr edgeCounts_;
  float gridAspect_;
  SurfaceGeometry* surfaceGeometry_;
  TrainParams trainParams_;
  index_t numCoords_;
  size_t totalEdges_;

  /// Variables created internally.
  std::default_random_engine rng_;
  float left_;
  float right_;
  float bottom_;
  float top_;
  FloatArr gridPoints_;
  index_t gridWidth_;
  index_t gridHeight_;
};

#endif