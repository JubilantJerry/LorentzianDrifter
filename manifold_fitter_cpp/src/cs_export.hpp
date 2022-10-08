extern "C" {

#ifdef _WIN32
#  ifdef MODULE_API_EXPORTS
#    define MODULE_API __declspec(dllexport)
#  else
#    define MODULE_API __declspec(dllimport)
#  endif
#else
#  define MODULE_API
#endif

#include "stdint.h"

MODULE_API void* ConstructManifoldFitter();

MODULE_API void DestructManifoldFitter(void* fitter);

typedef void (*SurfaceGeometryFnCallback)(
	float*, int32_t, float*, int32_t, int32_t, float*, int32_t);

MODULE_API void UseSurfaceGeometryFn(
	void* fitter, SurfaceGeometryFnCallback fn);

}