#include <iostream>
#include <cmath>
#include <vector>
#include "manifold_fitter.hpp"

constexpr float kPiFloat = static_cast<float>(M_PI);


void SphereCoordDiffs(const FloatArr leftCoords, const FloatArr rightCoords,
                      size_t batchSize, FloatArr dest) {
  for (size_t i = 0; i < batchSize; i++) {
    float diffPhi = leftCoords[2 * i] - rightCoords[2 * i];
    float diff_theta = leftCoords[2 * i + 1] - rightCoords[2 * i + 1];
    if (diff_theta > kPiFloat) diff_theta -= 2 * kPiFloat;
    if (diff_theta < -kPiFloat) diff_theta += 2 * kPiFloat;
    dest[2 * i] = diffPhi;
    dest[2 * i + 1] = diff_theta;
  }
}


void SphereMetricRootQuadForm(const FloatArr coords, const FloatArr vectors,
                              size_t batchSize, FloatArr dest) {
  for (size_t i = 0; i < batchSize; i++) {
    float phi = coords[2 * i];
    float deltaPhi = vectors[2 * i];
    float scaledDelta_theta = sinf(phi) * vectors[2 * i + 1];
    dest[i] = sqrtf(
        deltaPhi * deltaPhi + scaledDelta_theta * scaledDelta_theta);
  }
}



void SquareGridYWrap(
    float left, float right, float bottom, float top,
    size_t xCount, size_t yCount, std::vector<float>& coords,
    std::vector<index_t>& graph, std::vector<index_t>& edgeCounts) {
  size_t rowCount = yCount;
  size_t totalSize = (xCount + 1) * rowCount;
  coords = std::vector<float>(2 * totalSize);
  graph = std::vector<index_t>();
  edgeCounts = std::vector<index_t>(totalSize);
  float xDiff = (right - left) / xCount;
  float yDiff = (top - bottom) / yCount;
  for (size_t i = 0; i < xCount + 1; i++) {
    for (size_t j = 0; j < yCount; j++) {
      size_t index = i * rowCount + j;
      size_t edgeCount = 0;
      if (i > 0 and j > 0) {
        graph.push_back(index - rowCount - 1);
        edgeCount += 1;
      }
      if (i > 0 and j == 0) {
        graph.push_back(index - 1);
        edgeCount += 1;
      }
      if (i > 0) {
        graph.push_back(index - rowCount);
        edgeCount += 1;
      }
      if (i > 0 and j < yCount - 1) {
        graph.push_back(index - rowCount + 1);
        edgeCount += 1;
      }
      if (i > 0 and j == yCount - 1) {
        graph.push_back(index - 2 * rowCount + 1);
        edgeCount += 1;
      }
      if (j > 0) {
        graph.push_back(index - 1);
        edgeCount += 1;
      }
      if (j == 0) {
        graph.push_back(index + rowCount - 1);
        edgeCount += 1;
      }
      if (j < yCount - 1) {
        graph.push_back(index + 1);
        edgeCount += 1;
      }
      if (j == yCount - 1) {
        graph.push_back(index - rowCount + 1);
        edgeCount += 1;
      }
      if (i < xCount and j > 0) {
        graph.push_back(index + rowCount - 1);
        edgeCount += 1;
      }
      if (i < xCount and j == 0) {
        graph.push_back(index + 2 * rowCount - 1);
        edgeCount += 1;
      }
      if (i < xCount) {
        graph.push_back(index + rowCount);
        edgeCount += 1;
      }
      if (i < xCount and j < yCount - 1) {
        graph.push_back(index + rowCount + 1);
        edgeCount += 1;
      }
      if (i < xCount and j == yCount - 1) {
        graph.push_back(index + 1);
        edgeCount += 1;
      }
      edgeCounts[index] = edgeCount;
      coords[2 * index] = left + i * xDiff;
      coords[2 * index + 1] = bottom + j * yDiff;
    }
  }
}


int main() {
  ManifoldFitter fitter;
  std::vector<float> coords;
  std::vector<index_t> graph;
  std::vector<index_t> edgeCounts;
  SquareGridYWrap(0.15, kPiFloat - 0.15, -kPiFloat, kPiFloat, 10, 20,
                  coords, graph, edgeCounts);
  fitter.LoadCoords(coords.data(), edgeCounts.size());
  fitter.LoadGraph(graph.data(), edgeCounts.data(),
                   edgeCounts.size(), graph.size());
  fitter.SetGridAspect(0.5f);
  SurfaceGeometry geometry = {
    .CoordDiffs = &SphereCoordDiffs,
    .MetricRootQuadForm = &SphereMetricRootQuadForm};
  fitter.SetGeometry(&geometry);
  ManifoldFitter::TrainParams trainParams = {
    .lr = 0.1f, .gamma = 0.2f, .initNoise = 0.1f,
    .supersampleLimit = 20, .supersampleMult = 1.5f,
    .lossEpsilon = 1.e-5f, .stagnantThres = 20,
    .smallGridLimit = 5, .smallGridIters = 100,
    .printIters = true, .printTotalIters = true
  };
  fitter.SetTrainParams(trainParams);
  std::vector<float> dest(edgeCounts.size() * 3);
  for (size_t i = 0; i < 1; i++) {
    fitter.Fit(dest.data());
  }
  printf("[");
  for (size_t i = 0; i < edgeCounts.size(); i++) {
    printf("[%f, %f, %f]", dest[3 * i], dest[3 * i + 1], dest[3 * i + 2]);
    if (i != edgeCounts.size() - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}