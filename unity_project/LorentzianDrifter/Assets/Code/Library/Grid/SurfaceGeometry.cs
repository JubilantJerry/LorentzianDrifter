class SurfaceGeometry
{
    public delegate void CoordDiffsFn(
        float[] left, float[] right, int batchSize, float[] dest);

    public delegate void MetricRootQuadFormFn(
        float[] coords, float[] vecs, int batchSize, float[] dest);

    public CoordDiffsFn CoordDiffs { get; }

    public MetricRootQuadFormFn MetricRootQuadForm { get; }
}