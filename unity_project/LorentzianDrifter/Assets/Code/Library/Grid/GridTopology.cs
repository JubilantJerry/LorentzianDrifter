public class GridTopology
{
    public int NumVertices
    {
        get => GraphEdgeCounts.Length;
    }

    public int[] GraphEdgeCounts { get; set; }

    public int[] Graph { get; set; }

    public int[] MeshTriangles { get; set; }
}