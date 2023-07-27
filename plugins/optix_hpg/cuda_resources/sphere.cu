#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>

#include "optix/random.h"
#include "optix/utils_device.h"
#include "perraydata.h"
#include "sphere.cuh"
#include "sphere.h"


// OptiX SDK and rtxpkd

namespace megamol {
namespace optix_hpg {
    namespace device {
        MM_OPTIX_INTERSECTION_KERNEL(sphere_intersect)() {
            kernel_sphere_intersect();
        }


        MM_OPTIX_CLOSESTHIT_KERNEL(sphere_closesthit)() {
            kernel_sphere_closest_hit();
        }


        MM_OPTIX_BOUNDS_KERNEL(sphere_bounds)(const void* geomData, box3f& primBounds, const unsigned int primID) {
            kernel_bounds(geomData, primBounds, primID);
        }
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
