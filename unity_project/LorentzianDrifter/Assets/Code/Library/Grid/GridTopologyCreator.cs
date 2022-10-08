public class GridTopologyCreator
{
    public static GridTopology GetDemo() {
        GridTopology result = new GridTopology();
        result.GraphEdgeCounts = new int[3] {2, 2, 2};
        result.Graph = new int[6] {1, 2, 0, 2, 0, 1};
        result.MeshTriangles = new int[6] {0, 1, 2, 0, 2, 1};
        return result;
    }
}
