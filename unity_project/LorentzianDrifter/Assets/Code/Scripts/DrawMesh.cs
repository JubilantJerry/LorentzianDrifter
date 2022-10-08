using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class DrawMesh : MonoBehaviour
{
    private Mesh _mesh;
    public Material material;

    // Start is called before the first frame update
    void Start()
    {
        GridTopology grid = GridTopologyCreator.GetDemo();
        _mesh = new Mesh();
        _mesh.vertices = new Vector3[3] {
            new Vector3(0, 1, 0), new Vector3(0, 0, 1), new Vector3(1, 0, 0)};
        _mesh.triangles = grid.MeshTriangles;
        GetComponent<MeshFilter>().mesh = _mesh;
        ManifoldFitter fitter = new ManifoldFitter();
        fitter.UseSurfaceGeometryFn((left, right, batchSize, dest) => {
            dest[0] = left[0] + right[0] + right[1] + batchSize;
        });
    }

    // Update is called once per frame
    void Update()
    {
    }
}
