#include "manifold_fitter.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>


#define USE_SIMD 1


void ManifoldFitter::Prepare(WorkingData& data) {
  // Finding the minimum and maximum coordinates.
  float* coords = coords_;
  size_t numCoords = numCoords_;
  assert(numCoords_ > 0);
  float left = coords[0];
  float right = coords[0];
  float bottom = coords[1];
  float top = coords[1];
  for (size_t i = 0; i < numCoords; ++i) {
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
  size_t maxGridWidth, maxGridHeight;
  float maxGridSize = 1.0f;
  float supersampleMult = trainParams_.supersampleMult;
  float supersampleLimit = static_cast<float>(
      trainParams_.supersampleLimit);
  while (maxGridSize < supersampleLimit) {
    maxGridSize = floor(maxGridSize * supersampleMult) + 1.0f;
  }
  if (gridAspect_ <= 1.0f) {
    maxGridHeight = static_cast<size_t>(maxGridSize);
    maxGridWidth = static_cast<size_t>(maxGridSize * gridAspect_  + 0.5f);
  } else {
    maxGridWidth = static_cast<size_t>(maxGridSize);
    maxGridHeight = static_cast<size_t>(maxGridSize / gridAspect_ + 0.5f);
  }
  free(gridPoints_);
  // We allocate four floats per point even though we only need 3D points,
  // so that the SIMD loads will always be 16 byte aligned. The working set
  // is likely cache resident so the extra memory usage is fine.
  size_t gridPointsArrSize =
    (maxGridWidth + 1) * (maxGridHeight + 1) * 4 * sizeof(float);
  gridPoints_ = static_cast<FloatArr>(CacheLineAlloc(gridPointsArrSize));
  data.gridPointsMove = static_cast<FloatArr>(
    CacheLineAlloc(gridPointsArrSize));
  data.gridPointsCopy = static_cast<FloatArr>(
    CacheLineAlloc(gridPointsArrSize));

  // Allocate the interpolated point and grad arrays.
  size_t interpolatedPointsArrSize = numCoords_ * 4 * sizeof(float);
  data.interpolatedPoints = static_cast<FloatArr>(
      CacheLineAlloc(interpolatedPointsArrSize));
  data.interpolatedPointsGrad = static_cast<FloatArr>(
      CacheLineAlloc(interpolatedPointsArrSize));

  // Initializing the bilinear grid.
  gridWidth_ = 1;
  gridHeight_ = 1;
  float* gridPoints = gridPoints_;
  std::normal_distribution<float> normal(0.0f, trainParams_.initNoise);
  gridPoints[0 * (2 * 4) + 0 * 4 + 1] = left + normal(rng_);
  gridPoints[0 * (2 * 4) + 0 * 4 + 2] = bottom + normal(rng_);
  gridPoints[0 * (2 * 4) + 0 * 4 + 3] = normal(rng_);
  gridPoints[0 * (2 * 4) + 1 * 4 + 1] = left + normal(rng_);
  gridPoints[0 * (2 * 4) + 1 * 4 + 2] = top + normal(rng_);
  gridPoints[0 * (2 * 4) + 1 * 4 + 3] = normal(rng_);
  gridPoints[1 * (2 * 4) + 0 * 4 + 1] = right + normal(rng_);
  gridPoints[1 * (2 * 4) + 0 * 4 + 2] = bottom + normal(rng_);
  gridPoints[1 * (2 * 4) + 0 * 4 + 3] = normal(rng_);
  gridPoints[1 * (2 * 4) + 1 * 4 + 1] = right + normal(rng_);
  gridPoints[1 * (2 * 4) + 1 * 4 + 2] = top + normal(rng_);
  gridPoints[1 * (2 * 4) + 1 * 4 + 3] = normal(rng_);

  // Allocating the gridCorners and related.
  data.coordGridCorners = static_cast<IndexTArr>(
      CacheLineAlloc(numCoords_ * sizeof(index_t)));
  data.coordGridSurpluses = static_cast<FloatArr>(
      CacheLineAlloc(numCoords_ * 2 * sizeof(float)));

  // Finding the other vertex for each graph edge.
  size_t totalEdges = totalEdges_;
  index_t* edgeCounts = edgeCounts_;
  data.graphCenters = static_cast<IndexTArr>(
    CacheLineAlloc(totalEdges * sizeof(index_t)));
  for (size_t i = 0, edgeIndex = 0; i < numCoords; ++i) {
    for (size_t j = 0; j < edgeCounts[i]; ++j, ++edgeIndex) {
      data.graphCenters[edgeIndex] = i;
    }
  }

  // Finding the target lengths of graph edges.
  index_t* graphCenters = data.graphCenters;
  index_t* graph = graph_;
  FloatArr coordCenters = static_cast<FloatArr>(
    CacheLineAlloc(totalEdges * 2 * sizeof(float)));
  FloatArr coordNeighbors = static_cast<FloatArr>(
    CacheLineAlloc(totalEdges * 2 * sizeof(float)));
  FloatArr diffVecs = static_cast<FloatArr>(
    CacheLineAlloc(totalEdges * 2 * sizeof(float)));
  FloatArr edgeCenters = static_cast<FloatArr>(
    CacheLineAlloc(totalEdges * 2 * sizeof(float)));
  data.targetDistances = static_cast<FloatArr>(
    CacheLineAlloc(totalEdges * sizeof(float)));
  for (size_t i = 0; i < totalEdges; ++i) {
    size_t center = graphCenters[i];
    size_t neighbor = graph[i];
    coordCenters[2 * i] = coords[2 * center];
    coordCenters[2 * i + 1] = coords[2 * center + 1];
    coordNeighbors[2 * i] = coords[2 * neighbor];
    coordNeighbors[2 * i + 1] = coords[2 * neighbor + 1];
  }
  surfaceGeometry_->CoordDiffs(
      coordNeighbors, coordCenters, totalEdges, diffVecs);
  for (size_t i = 0; i < totalEdges; ++i) {
    float diffVecX = diffVecs[2 * i];
    float diffVecY = diffVecs[2 * i + 1];
    size_t center = graphCenters[i];
    edgeCenters[2 * i] = coords[2 * center] + diffVecX * 0.5f;
    edgeCenters[2 * i + 1] = coords[2 * center + 1] + diffVecY * 0.5f;
  }
  surfaceGeometry_->MetricRootQuadForm(
    edgeCenters, diffVecs, totalEdges, data.targetDistances);
  free(coordCenters);
  free(coordNeighbors);
  free(diffVecs);
  free(edgeCenters);

  // Allocating the scratchpad.
  size_t scratchpadSize = totalEdges * 4 * sizeof(float);
  data.scratchpad = static_cast<FloatArr>(
    CacheLineAlloc(scratchpadSize));
}


void ManifoldFitter::PrepareIteration(WorkingData& data) {
  // Initializing the grid corners for the coordinates.
  float* coords = coords_;
  index_t* coordGridCorners = data.coordGridCorners;
  float* coordGridSurpluses = data.coordGridSurpluses;
  float left = left_;
  float right = right_;
  float bottom = bottom_;
  float top = top_;
  float invWidth =  1.0f / (right - left);
  float invHeight = 1.0f / (top - bottom);
  size_t gridWidth = gridWidth_;
  size_t gridHeight = gridHeight_;
  size_t numCoords = numCoords_;
  for (size_t i = 0; i < numCoords; ++i) {
      float x = coords[2 * i];
      float y = coords[2 * i + 1];
      float xFrac = (x - left) * invWidth;
      float yFrac = (y - bottom) * invHeight;
      float xSurplus = xFrac * gridWidth;
      size_t xInd = static_cast<size_t>(xSurplus);
      if (xInd < 0) xInd = 0;
      if (xInd > gridWidth - 1) xInd = gridWidth - 1;
      xSurplus = xSurplus - static_cast<float>(xInd);
      float ySurplus = yFrac * gridHeight;
      size_t yInd = static_cast<size_t>(ySurplus);
      if (yInd < 0) yInd = 0;
      if (yInd > gridHeight - 1) yInd = gridHeight - 1;
      ySurplus = ySurplus - static_cast<float>(yInd);
      coordGridCorners[i] = 4 * (xInd * (gridHeight + 1) + yInd);
      coordGridSurpluses[2 * i] = xSurplus;
      coordGridSurpluses[2 * i + 1] = ySurplus;
  }

  // Reset the optimizer momentum.
  size_t gridPointsArrSize =
    (gridWidth + 1) * (gridHeight + 1) * 4 * sizeof(float);
  memset(data.gridPointsMove, 0, gridPointsArrSize);
}


float ManifoldFitter::RunIteration(WorkingData& data) {
  // Compute the interpolated points.
  float* gridPoints = gridPoints_;
  size_t gridHeightP1 = gridHeight_ + 1;
  index_t* coordGridCorners = data.coordGridCorners;
  float* coordGridSurpluses = data.coordGridSurpluses;
  float* interpolatedPoints = data.interpolatedPoints;
  float* interpolatedPointsGrad = data.interpolatedPointsGrad;
  size_t numCoords = numCoords_;
  double loss = 0.0f;

  #if USE_SIMD
    __m128 zeroV = _mm_set1_ps(0.0f);
  #endif
  for (size_t i = 0; i < numCoords; ++i) {
    float xSurplus = coordGridSurpluses[2 * i];
    float ySurplus = coordGridSurpluses[2 * i + 1];
    float xSurplusOpp = 1.0f - xSurplus;
    float ySurplusOpp = 1.0f - ySurplus;
    float topRightMult = xSurplus * ySurplus;
    float topLeftMult = xSurplusOpp * ySurplus;
    float botRightMult = xSurplus * ySurplusOpp;
    float botLeftMult = xSurplusOpp * ySurplusOpp;
    size_t coordGridCorner = coordGridCorners[i];
    size_t botLeftIndex = coordGridCorner;
    size_t botRightIndex = coordGridCorner + 4 * gridHeightP1;
    size_t topLeftIndex = coordGridCorner + 4;
    size_t topRightIndex = botRightIndex + 4;
    #if USE_SIMD
      _mm_store_ps(&interpolatedPointsGrad[4 * i], zeroV);
      __m128 botLeftV = _mm_load_ps(&gridPoints[botLeftIndex]);
      __m128 topLeftV = _mm_load_ps(&gridPoints[topLeftIndex]);
      __m128 botRightV = _mm_load_ps(&gridPoints[botRightIndex]);
      __m128 topRightV = _mm_load_ps(&gridPoints[topRightIndex]);
      __m128 topRightMultV = _mm_set1_ps(topRightMult);
      __m128 topLeftMultV = _mm_set1_ps(topLeftMult);
      __m128 botRightMultV = _mm_set1_ps(botRightMult);
      __m128 botLeftMultV = _mm_set1_ps(botLeftMult);
      __m128 interpolated = _mm_add_ps(
        _mm_fmadd_ps(
          topRightMultV, topRightV,
          _mm_mul_ps(topLeftMultV, topLeftV)),
        _mm_fmadd_ps(
          botRightMultV, botRightV,
          _mm_mul_ps(botLeftMultV, botLeftV)));
      _mm_store_ps(&interpolatedPoints[4 * i], interpolated);
    #else
      interpolatedPointsGrad[4 * i + 1] = 0.0f;
      interpolatedPointsGrad[4 * i + 2] = 0.0f;
      interpolatedPointsGrad[4 * i + 3] = 0.0f;
      interpolatedPoints[4 * i + 1] =
        topRightMult * gridPoints[topRightIndex + 1] +
        topLeftMult * gridPoints[topLeftIndex + 1] +
        botRightMult * gridPoints[botRightIndex + 1] +
        botLeftMult * gridPoints[botLeftIndex + 1];
      interpolatedPoints[4 * i + 2] =
        topRightMult * gridPoints[topRightIndex + 2] +
        topLeftMult * gridPoints[topLeftIndex + 2] +
        botRightMult * gridPoints[botRightIndex + 2] +
        botLeftMult * gridPoints[botLeftIndex + 2];
      interpolatedPoints[4 * i + 3] =
        topRightMult * gridPoints[topRightIndex + 3] +
        topLeftMult * gridPoints[topLeftIndex + 3] +
        botRightMult * gridPoints[botRightIndex + 3] +
        botLeftMult * gridPoints[botLeftIndex + 3];
    #endif
  }

  // Compute the interpolated point gradients.
  float* targetDistances = data.targetDistances;
  size_t totalEdges = totalEdges_;
  index_t* graphCenters = data.graphCenters;
  index_t* graph = graph_;
  float* scratchpad = data.scratchpad;
  #if USE_SIMD
    __m128 oneV = _mm_set1_ps(1.0f);
    __m128 threeV = _mm_set1_ps(3.0f);
    __m128 halfV = _mm_set1_ps(0.5f);
    // Unroll the loop across edges by four.
    for (size_t i = 0; i + 3 < totalEdges; i += 4) {
      // Fill a 128 bit XMM register with values of trueDistanceSq.
      size_t center = graphCenters[i];
      size_t neighbor = graph[i];
      __m128 centerV = _mm_load_ps(&interpolatedPoints[4 * center]);
      __m128 neighborV = _mm_load_ps(&interpolatedPoints[4 * neighbor]);
      __m128 diff_0V = _mm_sub_ps(neighborV, centerV);
      __m128 trueDistanceSqV = _mm_dp_ps(diff_0V, diff_0V, 0xE1);
      center = graphCenters[i + 1];
      neighbor = graph[i + 1];
      centerV = _mm_load_ps(&interpolatedPoints[4 * center]);
      neighborV = _mm_load_ps(&interpolatedPoints[4 * neighbor]);
      __m128 diff_1V = _mm_sub_ps(neighborV, centerV);
      trueDistanceSqV = _mm_add_ps(
          trueDistanceSqV, _mm_dp_ps(diff_1V, diff_1V, 0xE2));
      center = graphCenters[i + 2];
      neighbor = graph[i + 2];
      centerV = _mm_load_ps(&interpolatedPoints[4 * center]);
      neighborV = _mm_load_ps(&interpolatedPoints[4 * neighbor]);
      __m128 diff_2V = _mm_sub_ps(neighborV, centerV);
      trueDistanceSqV = _mm_add_ps(
          trueDistanceSqV, _mm_dp_ps(diff_2V, diff_2V, 0xE4));
      center = graphCenters[i + 3];
      neighbor = graph[i + 3];
      centerV = _mm_load_ps(&interpolatedPoints[4 * center]);
      neighborV = _mm_load_ps(&interpolatedPoints[4 * neighbor]);
      __m128 diff_3V = _mm_sub_ps(neighborV, centerV);
      trueDistanceSqV = _mm_add_ps(
          trueDistanceSqV, _mm_dp_ps(diff_3V, diff_3V, 0xE8));
      // Compute the inverse square root of the four values with rsqrt and
      // one Newton-Raphson iteration. This is faster than the direct method
      // with almost the same precision.
      __m128 trueDistanceInvV = _mm_rsqrt_ps(trueDistanceSqV);
      __m128 multTermV = _mm_mul_ps(_mm_mul_ps(
          trueDistanceSqV, trueDistanceInvV), trueDistanceInvV);
      trueDistanceInvV = _mm_mul_ps(
        _mm_mul_ps(halfV, trueDistanceInvV),
        _mm_sub_ps(threeV, multTermV));
      // Use the four values to compute four vectors for the scratchpad.
      __m128 targetDistanceV = _mm_load_ps(&targetDistances[i]);
      __m128 gradMultV = _mm_sub_ps(
        oneV, _mm_mul_ps(targetDistanceV, trueDistanceInvV));
      __m128 diff_termSqV = _mm_mul_ps(_mm_mul_ps(
          gradMultV, gradMultV), trueDistanceSqV);
      __m128 gradMultBcastV = _mm_shuffle_ps(
          gradMultV, gradMultV, 0x00);
      __m128 gradUpdateV = _mm_mul_ps(gradMultBcastV, diff_0V);
      __m128 saveV = _mm_move_ss(
          gradUpdateV, _mm_shuffle_ps(diff_termSqV, diff_termSqV, 0x00));
      _mm_store_ps(&scratchpad[4 * i], saveV);
      gradMultBcastV = _mm_shuffle_ps(
          gradMultV, gradMultV, 0x55);
      gradUpdateV = _mm_mul_ps(gradMultBcastV, diff_1V);
      saveV = _mm_move_ss(
          gradUpdateV, _mm_shuffle_ps(diff_termSqV, diff_termSqV, 0x01));
      _mm_store_ps(&scratchpad[4 * i + 4], saveV);
      gradMultBcastV = _mm_shuffle_ps(
          gradMultV, gradMultV, 0xAA);
      gradUpdateV = _mm_mul_ps(gradMultBcastV, diff_2V);
      saveV = _mm_move_ss(
          gradUpdateV, _mm_shuffle_ps(diff_termSqV, diff_termSqV, 0x02));
      _mm_store_ps(&scratchpad[4 * i + 8], saveV);
      gradMultBcastV = _mm_shuffle_ps(
          gradMultV, gradMultV, 0xFF);
      gradUpdateV = _mm_mul_ps(gradMultBcastV, diff_3V);
      saveV = _mm_move_ss(
          gradUpdateV, _mm_shuffle_ps(diff_termSqV, diff_termSqV, 0x03));
      _mm_store_ps(&scratchpad[4 * i + 12], saveV);
    }
    // Use the scalar implementation for any remaining edges.
    for (size_t i = (totalEdges / 4) * 4; i < totalEdges; ++i) {
  #else
    for (size_t i = 0; i < totalEdges; ++i) {
  #endif
      size_t center = graphCenters[i];
      size_t neighbor = graph[i];
      float centerX = interpolatedPoints[4 * center + 1];
      float centerY = interpolatedPoints[4 * center + 2];
      float centerZ = interpolatedPoints[4 * center + 3];
      float neighborX = interpolatedPoints[4 * neighbor + 1];
      float neighborY = interpolatedPoints[4 * neighbor + 2];
      float neighborZ = interpolatedPoints[4 * neighbor + 3];
      float diffX = neighborX - centerX;
      float diffY = neighborY - centerY;
      float diffZ = neighborZ - centerZ;
      float trueDistanceSq =
          diffX * diffX + diffY * diffY + diffZ * diffZ;
      float trueDistanceInv = 1.0f / sqrtf(trueDistanceSq);
      float targetDistance = targetDistances[i];
      float gradMult = (1.0f - targetDistance * trueDistanceInv);
      float diff_termSq = gradMult * gradMult * trueDistanceSq;
      scratchpad[4 * i] = diff_termSq;
      scratchpad[4 * i + 1] = gradMult * diffX;
      scratchpad[4 * i + 2] = gradMult * diffY;
      scratchpad[4 * i + 3] = gradMult * diffZ;
  }
  for (size_t i = 0; i < totalEdges; ++i) {
    #if USE_SIMD
      size_t center = graphCenters[i];
      size_t neighbor = graph[i];
      __m128 gradUpdateV = _mm_load_ps(&scratchpad[4 * i]);
      __m128 centerV = _mm_load_ps(&interpolatedPointsGrad[4 * center]);
      __m128 neighborV = _mm_load_ps(&interpolatedPointsGrad[4 * neighbor]);
      loss += _mm_cvtss_f32(gradUpdateV);
      centerV = _mm_sub_ps(centerV, gradUpdateV);
      neighborV = _mm_add_ps(neighborV, gradUpdateV);
      _mm_store_ps(&interpolatedPointsGrad[4 * center], centerV);
      _mm_store_ps(&interpolatedPointsGrad[4 * neighbor], neighborV);
    #else
      size_t center = graphCenters[i];
      size_t neighbor = graph[i];
      float gradUpdateX = scratchpad[4 * i + 1];
      float gradUpdateY = scratchpad[4 * i + 2];
      float gradUpdateZ = scratchpad[4 * i + 3];
      loss += scratchpad[4 * i];
      interpolatedPointsGrad[4 * center + 1] -= gradUpdateX;
      interpolatedPointsGrad[4 * center + 2] -= gradUpdateY;
      interpolatedPointsGrad[4 * center + 3] -= gradUpdateZ;
      interpolatedPointsGrad[4 * neighbor + 1] += gradUpdateX;
      interpolatedPointsGrad[4 * neighbor + 2] += gradUpdateY;
      interpolatedPointsGrad[4 * neighbor + 3] += gradUpdateZ;
    #endif
  }

  // Use the gradient to update the parameters and momentum.
  float* gridPointsMove = data.gridPointsMove;
  float gammaMult = 1.0f - trainParams_.gamma;
  float gradMult = trainParams_.lr * trainParams_.gamma / gammaMult;
  for (size_t i = 0; i < numCoords; ++i) {
      float xSurplus = coordGridSurpluses[2 * i];
      float ySurplus = coordGridSurpluses[2 * i + 1];
      float xSurplusOpp = 1.0f - xSurplus;
      float ySurplusOpp = 1.0f - ySurplus;
      size_t coordGridCorner = coordGridCorners[i];
      size_t botLeftIndex = coordGridCorner;
      size_t topLeftIndex = coordGridCorner + 4;
      size_t botRightIndex = coordGridCorner + 4 * gridHeightP1;
      size_t topRightIndex = coordGridCorner + 4 * gridHeightP1 + 4;
      float topRightMult = gradMult * xSurplus * ySurplus;
      float topLeftMult = gradMult * xSurplusOpp * ySurplus;
      float botRightMult = gradMult * xSurplus * ySurplusOpp;
      float botLeftMult = gradMult * xSurplusOpp * ySurplusOpp;
      #if USE_SIMD
        __m128 gradV = _mm_load_ps(&interpolatedPointsGrad[4 * i]);
        __m128 topRightV = _mm_load_ps(&gridPointsMove[topRightIndex]);
        __m128 topLeftV = _mm_load_ps(&gridPointsMove[topLeftIndex]);
        __m128 botRightV = _mm_load_ps(&gridPointsMove[botRightIndex]);
        __m128 botLeftV = _mm_load_ps(&gridPointsMove[botLeftIndex]);
        __m128 topRightMultV = _mm_set1_ps(topRightMult);
        __m128 topLeftMultV = _mm_set1_ps(topLeftMult);
        __m128 botRightMultV = _mm_set1_ps(botRightMult);
        __m128 botLeftMultV = _mm_set1_ps(botLeftMult);
        topRightV = _mm_fmadd_ps(topRightMultV, gradV, topRightV);
        topLeftV = _mm_fmadd_ps(topLeftMultV, gradV, topLeftV);
        botRightV = _mm_fmadd_ps(botRightMultV, gradV, botRightV);
        botLeftV = _mm_fmadd_ps(botLeftMultV, gradV, botLeftV);
        _mm_store_ps(&gridPointsMove[topRightIndex], topRightV);
        _mm_store_ps(&gridPointsMove[topLeftIndex], topLeftV);
        _mm_store_ps(&gridPointsMove[botRightIndex], botRightV);
        _mm_store_ps(&gridPointsMove[botLeftIndex], botLeftV);
      #else
        float gradX = interpolatedPointsGrad[4 * i + 1];
        float gradY = interpolatedPointsGrad[4 * i + 2];
        float gradZ = interpolatedPointsGrad[4 * i + 3];
        gridPointsMove[topRightIndex + 1] += topRightMult * gradX;
        gridPointsMove[topRightIndex + 2] += topRightMult * gradY;
        gridPointsMove[topRightIndex + 3] += topRightMult * gradZ;
        gridPointsMove[topLeftIndex + 1] += topLeftMult * gradX;
        gridPointsMove[topLeftIndex + 2] +=  topLeftMult * gradY;
        gridPointsMove[topLeftIndex + 3] += topLeftMult * gradZ;
        gridPointsMove[botRightIndex + 1] += botRightMult * gradX;
        gridPointsMove[botRightIndex + 2] += botRightMult * gradY;
        gridPointsMove[botRightIndex + 3] += botRightMult * gradZ;
        gridPointsMove[botLeftIndex + 1] += botLeftMult * gradX;
        gridPointsMove[botLeftIndex + 2] +=  botLeftMult * gradY;
        gridPointsMove[botLeftIndex + 3] += botLeftMult * gradZ;
      #endif
  }
  size_t totalGridPoints = (gridWidth_ + 1) * gridHeightP1;
  #if USE_SIMD
    __m128 gammaMultV = _mm_set1_ps(gammaMult);
  #endif
  for (size_t i = 0; i < totalGridPoints; ++i) {
    #if USE_SIMD
      __m128 moveV = _mm_mul_ps(
          gammaMultV, _mm_load_ps(&gridPointsMove[4 * i]));
      __m128 gridPointV = _mm_load_ps(&gridPoints[4 * i]);
      _mm_store_ps(&gridPointsMove[4 * i], moveV);
      gridPointV = _mm_sub_ps(gridPointV, moveV);
      _mm_store_ps(&gridPoints[4 * i], gridPointV);
    #else
      float moveX = gammaMult * gridPointsMove[4 * i + 1];
      float moveY = gammaMult * gridPointsMove[4 * i + 2];
      float moveZ = gammaMult * gridPointsMove[4 * i + 3];
      gridPointsMove[4 * i + 1] = moveX;
      gridPointsMove[4 * i + 2] = moveY;
      gridPointsMove[4 * i + 3] = moveZ;
      gridPoints[4 * i + 1] -= moveX;
      gridPoints[4 * i + 2] -= moveY;
      gridPoints[4 * i + 3] -= moveZ;
    #endif
  }
  loss *= 0.5;

  return loss;
}


void ManifoldFitter::Supersample(WorkingData& data, float supersampleMult) {
  // Copy the grid points to another buffer.
  float* gridPoints = gridPoints_;
  float* gridPointsCopy = data.gridPointsCopy;
  size_t gridWidth = gridWidth_;
  size_t gridHeight = gridHeight_;
  memcpy(gridPointsCopy, gridPoints,
         (gridHeight + 1) * (gridWidth + 1) * 4 * sizeof(float));

  // Compute the new grid dimensions.
  size_t newGridWidth = 0;
  size_t newGridHeight = 0;
  if (gridAspect_ <= 1.0f) {
    newGridHeight = static_cast<size_t>(gridHeight_ * supersampleMult) + 1;
    newGridWidth = static_cast<size_t>(
      newGridHeight * gridAspect_  + 0.5f);
  } else {
    newGridWidth = static_cast<size_t>(gridWidth_ * supersampleMult) + 1;
    newGridHeight = static_cast<size_t>(
      newGridWidth / gridAspect_  + 0.5f);
  }

  // Compute the new grid points.
  for (size_t i = 0, gridIndex = 0; i < newGridWidth + 1; i++) {
    float xFrac = static_cast<float>(i) / newGridWidth;
    for (size_t j = 0; j < newGridHeight + 1; j++, gridIndex++) {
      float yFrac = static_cast<float>(j) / newGridHeight;
      float xSurplus = xFrac * gridWidth;
      size_t xInd = static_cast<size_t>(xSurplus);
      if (xInd < 0) xInd = 0;
      if (xInd > gridWidth - 1) xInd = gridWidth - 1;
      xSurplus = xSurplus - static_cast<float>(xInd);
      float ySurplus = yFrac * gridHeight;
      size_t yInd = static_cast<size_t>(ySurplus);
      if (yInd < 0) yInd = 0;
      if (yInd > gridHeight - 1) yInd = gridHeight - 1;
      ySurplus = ySurplus - static_cast<float>(yInd);
      size_t coordGridCorner = 4 * (xInd * (gridHeight + 1) + yInd);
      float xSurplusOpp = 1.0f - xSurplus;
      float ySurplusOpp = 1.0f - ySurplus;
      size_t botLeftIndex = coordGridCorner;
      size_t topLeftIndex = coordGridCorner + 4;
      size_t botRightIndex = coordGridCorner + 4 * (gridHeight + 1);
      size_t topRightIndex = coordGridCorner + 4 * (gridHeight + 1) + 4;
      float topRightMult = xSurplus * ySurplus;
      float topLeftMult = xSurplusOpp * ySurplus;
      float botRightMult = xSurplus * ySurplusOpp;
      float botLeftMult = xSurplusOpp * ySurplusOpp;
      #if USE_SIMD
        __m128 botLeftV = _mm_load_ps(&gridPointsCopy[botLeftIndex]);
        __m128 topLeftV = _mm_load_ps(&gridPointsCopy[topLeftIndex]);
        __m128 botRightV = _mm_load_ps(&gridPointsCopy[botRightIndex]);
        __m128 topRightV = _mm_load_ps(&gridPointsCopy[topRightIndex]);
        __m128 topRightMultV = _mm_set1_ps(topRightMult);
        __m128 topLeftMultV = _mm_set1_ps(topLeftMult);
        __m128 botRightMultV = _mm_set1_ps(botRightMult);
        __m128 botLeftMultV = _mm_set1_ps(botLeftMult);
        __m128 interpolated = _mm_add_ps(
          _mm_fmadd_ps(
            topRightMultV, topRightV,
            _mm_mul_ps(topLeftMultV, topLeftV)),
          _mm_fmadd_ps(
            botRightMultV, botRightV,
            _mm_mul_ps(botLeftMultV, botLeftV)));
        _mm_store_ps(&gridPoints[4 * gridIndex], interpolated);
      #else
        gridPoints[4 * gridIndex + 1] =
          topRightMult * gridPointsCopy[topRightIndex + 1] +
          topLeftMult * gridPointsCopy[topLeftIndex + 1] +
          botRightMult * gridPointsCopy[botRightIndex + 1] +
          botLeftMult * gridPointsCopy[botLeftIndex + 1];
        gridPoints[4 * gridIndex + 2] =
          topRightMult * gridPointsCopy[topRightIndex + 2] +
          topLeftMult * gridPointsCopy[topLeftIndex + 2] +
          botRightMult * gridPointsCopy[botRightIndex + 2] +
          botLeftMult * gridPointsCopy[botLeftIndex + 2];
        gridPoints[4 * gridIndex + 3] =
          topRightMult * gridPointsCopy[topRightIndex + 3] +
          topLeftMult * gridPointsCopy[topLeftIndex + 3] +
          botRightMult * gridPointsCopy[botRightIndex + 3] +
          botLeftMult * gridPointsCopy[botLeftIndex + 3];
      #endif
    }
  }
  gridWidth_ = newGridWidth;
  gridHeight_ = newGridHeight;
}


void ManifoldFitter::Cleanup(WorkingData& data) {
  free(data.coordGridCorners);
  free(data.coordGridSurpluses);
  free(data.graphCenters);
  free(data.targetDistances);
  free(data.interpolatedPoints);
  free(data.interpolatedPointsGrad);
  free(data.gridPointsMove);
  free(data.gridPointsCopy);
  free(data.scratchpad);
}


void ManifoldFitter::Interpolate(FloatArr newCoords, index_t numNewCoords,
                                 FloatArr dest) {
  // Compute the new grid points.
  float left = left_;
  float right = right_;
  float bottom = bottom_;
  float top = top_;
  size_t gridWidth = gridWidth_;
  size_t gridHeight = gridHeight_;
  float* gridPoints = gridPoints_;
  float invWidth =  1.0f / (right - left);
  float invHeight = 1.0f / (top - bottom);
  for (size_t i = 0; i < numNewCoords; i++) {
    float x = newCoords[2 * i];
    float y = newCoords[2 * i + 1];
    float xFrac = (x - left) * invWidth;
    float yFrac = (y - bottom) * invHeight;
    float xSurplus = xFrac * gridWidth;
    size_t xInd = static_cast<size_t>(xSurplus);
    if (xInd < 0) xInd = 0;
    if (xInd > gridWidth - 1) xInd = gridWidth - 1;
    xSurplus = xSurplus - static_cast<float>(xInd);
    float ySurplus = yFrac * gridHeight;
    size_t yInd = static_cast<size_t>(ySurplus);
    if (yInd < 0) yInd = 0;
    if (yInd > gridHeight - 1) yInd = gridHeight - 1;
    ySurplus = ySurplus - static_cast<float>(yInd);
    size_t coordGridCorner = 4 * (xInd * (gridHeight + 1) + yInd);
    float xSurplusOpp = 1.0f - xSurplus;
    float ySurplusOpp = 1.0f - ySurplus;
    size_t botLeftIndex = coordGridCorner;
    size_t topLeftIndex = coordGridCorner + 4;
    size_t botRightIndex = coordGridCorner + 4 * (gridHeight + 1);
    size_t topRightIndex = coordGridCorner + 4 * (gridHeight + 1) + 4;
    dest[3 * i] =
      xSurplus * ySurplus * gridPoints[topRightIndex + 1] +
      xSurplusOpp * ySurplus * gridPoints[topLeftIndex + 1] +
      xSurplus * ySurplusOpp * gridPoints[botRightIndex + 1] +
      xSurplusOpp * ySurplusOpp * gridPoints[botLeftIndex + 1];
    dest[3 * i + 1] =
      xSurplus * ySurplus * gridPoints[topRightIndex + 2] +
      xSurplusOpp * ySurplus * gridPoints[topLeftIndex + 2] +
      xSurplus * ySurplusOpp * gridPoints[botRightIndex + 2] +
      xSurplusOpp * ySurplusOpp * gridPoints[botLeftIndex + 2];
    dest[3 * i + 2] =
      xSurplus * ySurplus * gridPoints[topRightIndex + 3] +
      xSurplusOpp * ySurplus * gridPoints[topLeftIndex + 3] +
      xSurplus * ySurplusOpp * gridPoints[botRightIndex + 3] +
      xSurplusOpp * ySurplusOpp * gridPoints[botLeftIndex + 3];
  }
}


void ManifoldFitter::Fit(FloatArr dest) {
  WorkingData data;
  Prepare(data);
  float lossEpsilon = trainParams_.lossEpsilon;
  size_t smallGridLimit = trainParams_.smallGridLimit;
  size_t smallGridIters = trainParams_.smallGridIters;
  size_t stagnantThres = trainParams_.stagnantThres;
  float supersampleMult = trainParams_.supersampleMult;
  size_t supersampleLimit = trainParams_.supersampleLimit;
  size_t totalIters = 0;
  bool printIters = trainParams_.printIters;
  bool print_totalIters = trainParams_.printTotalIters;

  while (true) {
    size_t stagnantIters = 0;
    float prevLoss = 0.0f;
    bool hasPrevLoss = false;
    size_t iters = 0;
    PrepareIteration(data);

    while (true) {
      float loss = RunIteration(data);
      if (hasPrevLoss) {
        float lossDiff = loss - prevLoss;
        if (lossDiff < 0.0f) lossDiff = -lossDiff;
        if (lossDiff < lossEpsilon) {
          ++stagnantIters;
        } else {
          stagnantIters = 0;
        }
      }
      ++iters;
      ++totalIters;
      if (printIters) {
        printf("%0.3f ", loss);
        fflush(stdout);
      }
      bool exitDue_toSmall = (
          gridWidth_ <= smallGridLimit &&
          gridHeight_ <= smallGridLimit &&
          iters >= smallGridIters);
      bool exitDue_toStagnant = (
          (gridWidth_ > smallGridLimit ||
           gridHeight_ > smallGridLimit) &&
          stagnantIters >= stagnantThres);
      if (exitDue_toSmall || exitDue_toStagnant) {
        break;
      }
      hasPrevLoss = true;
      prevLoss = loss;
    }

    if (printIters) printf("\n");
    if (gridWidth_ >= supersampleLimit ||
        gridHeight_ >= supersampleLimit) {
      break;
    }
    if (gridWidth_ <= smallGridLimit &&
        gridHeight_ <= smallGridLimit) {
      Supersample(data, 1.0f);
    } else {
      Supersample(data, supersampleMult);
    }
  }
  if (print_totalIters) printf("Total iterations: %lu\n", totalIters);
  Cleanup(data);
  Interpolate(coords_, numCoords_, dest);
}