using System;
using System.Runtime.InteropServices;

class ManifoldFitter
{
    public ManifoldFitter() => _handle = ConstructManifoldFitterImpl();

    ~ManifoldFitter()
    {
        DestructManifoldFitterImpl(_handle);
    }

    public void UseSurfaceGeometryFn(SurfaceGeometry.CoordDiffsFn fn)
    {
        UseSurfaceGeometryFnImpl(
            _handle,
            (arg1, size1, arg2, size2, batchSize, dest, destSize) =>
            {
                fn(arg1, arg2, batchSize, dest);
            });
    }

    private delegate void SurfaceGeometryFnCallback(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex=1)]
        [In] float[] arg1,
        int size1,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex=3)]
        [In] float[] arg2,
        int size2,
        int batchSize,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex=6)]
        [Out] float[] dest,
        int destSize
    );

    [DllImport(@"mfit_lib.so", EntryPoint = "ConstructManifoldFitter")]
    private static extern IntPtr ConstructManifoldFitterImpl();

    [DllImport(@"mfit_lib.so", EntryPoint = "DestructManifoldFitter")]
    private static extern void DestructManifoldFitterImpl(IntPtr fitter);

    [DllImport(@"mfit_lib.so", EntryPoint = "UseSurfaceGeometryFn")]
    private static extern void UseSurfaceGeometryFnImpl(
        IntPtr fitter, SurfaceGeometryFnCallback fn);

    private IntPtr _handle;
}