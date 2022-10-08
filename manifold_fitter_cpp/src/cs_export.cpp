#include "cs_export.hpp"
#include "manifold_fitter.hpp"

void* ConstructManifoldFitter() {
	std::cerr << "Construct" << std::endl;
	return static_cast<void*>(new ManifoldFitter());
}

void DestructManifoldFitter(void* fitter) {
	std::cerr << "Destruct" << std::endl;
	delete static_cast<ManifoldFitter*>(fitter);
}

void UseSurfaceGeometryFn(void* fitter, SurfaceGeometryFnCallback fn) {
	float arg1 = 0.3;
	float arg2[2] = {1.0, 2.1};
	float dest;
	SurfaceGeometry geometry = {
		[&](FloatArr arg1, FloatArr arg2, size_t batchSize, FloatArr dest) {
			fn(arg1, 1, arg2, 2, batchSize, dest, 1);
		},
		[](FloatArr, FloatArr, size_t, FloatArr) {}};
	geometry.CoordDiffs(&arg1, arg2, 3, &dest);
	std::cerr << "Computed result is " << dest << std::endl;
}