#include <cuda_runtime.h>
#include <optix_device.h>

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>
#include <owl/owl_device.h>

#include "intersect.h"
#include "perraydata.h"
#include "floatcomp.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;

struct FloatCompStackEntry {
    float t0, t1;
    unsigned int nodeID;
};

struct FloatCompStackEntryDep {
    float t0, t1;
    unsigned int nodeID;
    vec3f refPos;
};

inline __device__ vec3f getParticle(
    QTParticle_e5m15 const& data, int& dim, char const* exp_x_lu, char const* exp_y_lu, char const* exp_z_lu) {
    vec3f pos;

    unsigned int x = 0;
    char exp_x = exp_x_lu[data.exp_x];
    x += ((int) exp_x + 127u) << 23;
    x += (((unsigned int) data.m_x) << QTParticle_e5m15::offset);

    unsigned int y = 0;
    char exp_y = exp_y_lu[data.exp_y];
    y += ((int) exp_y + 127u) << 23;
    y += (((unsigned int) data.m_y) << QTParticle_e5m15::offset);

    unsigned int z = 0;
    char exp_z = exp_z_lu[data.exp_z];
    z += ((int) exp_z + 127u) << 23;
    z += (((unsigned int) data.m_z) << QTParticle_e5m15::offset);

    pos.x = __uint_as_float(x);
    pos.y = __uint_as_float(y);
    pos.z = __uint_as_float(z);

    if (data.dim_x)
        dim = 0;
    if (data.dim_y)
        dim = 1;
    if (data.dim_z)
        dim = 2;

    return pos;
}
inline __device__ int getDim(QTParticle_e5m15 const& data) {
    if (data.dim_x)
        return 0;
    if (data.dim_y)
        return 1;
    if (data.dim_z)
        return 2;
}

inline __device__ vec3f getParticle(QTParticle_e5m15d const& data, bool left_child, int sep_dim, int& dim,
    char const* exp_x_lu, char const* exp_y_lu, char const* exp_z_lu) {

    dim = (data.dim_b << 1) + data.dim_a;

    vec3f pos;

    unsigned int sign_x = sep_dim == 0 ? left_child : data.sign_a;
    unsigned int sign_y = sep_dim == 1 ? left_child : (sep_dim == 0 ? data.sign_a : data.sign_b);
    unsigned int sign_z = sep_dim == 2 ? left_child : data.sign_b;

    unsigned int x = 0;
    x += (sign_x) << 31;
    char exp_x = exp_x_lu[data.exp_x];
    x += ((int) exp_x + 127u) << 23;
    x += (((unsigned int) data.m_x) << QTParticle_e5m15d::offset);

    unsigned int y = 0;
    y += (sign_y) << 31;
    char exp_y = exp_y_lu[data.exp_y];
    y += ((int) exp_y + 127u) << 23;
    y += (((unsigned int) data.m_y) << QTParticle_e5m15d::offset);

    unsigned int z = 0;
    z += (sign_z) << 31;
    char exp_z = exp_z_lu[data.exp_z];
    z += ((int) exp_z + 127u) << 23;
    z += (((unsigned int) data.m_z) << QTParticle_e5m15d::offset);

    pos.x = __uint_as_float(x);
    pos.y = __uint_as_float(y);
    pos.z = __uint_as_float(z);

    return pos;
}
inline __device__ int getDim(QTParticle_e5m15d const& data) {
    return (data.dim_b << 1) + data.dim_a;
}

template<typename BUF_TYPE, int BEXP = BUF_TYPE::exp>
void __device__ traverse_dep(BUF_TYPE const* buffer) {
    const int treeletID = optixGetPrimitiveIndex();
    const auto& self = owl::getProgramData<FloatCompGeomData>();
    const auto& treelet = self.treeletBuffer[treeletID];
    PerRayData& prd = owl::getPRD<PerRayData>();
    //auto const localTables = self.localTables;

    int const num_idx = powf(2, BEXP);

    char const* exp_x_lu = self.expXBuffer;
    char const* exp_y_lu = self.expYBuffer;
    char const* exp_z_lu = self.expZBuffer;

    if (self.use_localtables > 0) {
        exp_x_lu += treeletID * num_idx;
        exp_y_lu += treeletID * num_idx;
        exp_z_lu += treeletID * num_idx;
    }

    //#ifdef LOCAL_TABLES
    //    if (treelet.has_local_tables) {
    //        exp_x_lu = treelet.exp_x;
    //        exp_y_lu = treelet.exp_y;
    //        exp_z_lu = treelet.exp_z;
    //    }
    //#endif


    //const auto& test_ref = self.qtpBuffer[0];

    auto const ray = owl::Ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

    const int begin = treelet.begin;
    const int size = treelet.end - begin;
    {
        float t0, t1;
        if (!clipToBounds(ray, treelet.bounds, t0, t1))
            return;


        int nodeID = 0;
        float tmp_hit_t = t1;
        int tmp_hit_primID = -1;

        enum { STACK_DEPTH = 12 };

        FloatCompStackEntryDep stackBase[STACK_DEPTH];
        FloatCompStackEntryDep* stackPtr = stackBase;

        const int dir_sign[3] = {ray.direction.x < 0.f, ray.direction.y < 0.f, ray.direction.z < 0.f};
        const float org[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
        const float rdir[3] = {
            (fabsf(ray.direction.x) <= 1e-8f) ? 1e8f : 1.f / ray.direction.x,
            (fabsf(ray.direction.y) <= 1e-8f) ? 1e8f : 1.f / ray.direction.y,
            (fabsf(ray.direction.z) <= 1e-8f) ? 1e8f : 1.f / ray.direction.z,
        };


        vec3f pos;
        int dim;
        vec3f refPos = treelet.basePos;


        vec3f tmp_hit_pos;

        //const float radius = self.radius;

        //unsigned int baseOffset = 0;
        /*if (localTables) {
            baseOffset = 1;
        }*/

        //auto const* buffer = (QTParticle_e4m16d const*)self.qtpBuffer;
        while (1) {
            // while we have anything to traverse ...

            while (1) {
                // while we can go down

                const int particleID = nodeID + begin;

                //auto test = self.qtpBuffer[particleID];

                //getParticle(treelet, self.qtpBuffer[particleID], dim, pos, exp_x_lu, exp_y_lu, exp_z_lu);

                /*if (nodeID == 0) {
                    pos = refPos;
                    dim = getDim(buffer[particleID]);
                } else*/
                {
                    auto const parentID = parent(nodeID) + begin;

                    //unsigned int offset = treeletID * num_idx * baseOffset;
                    pos = getParticle(buffer[particleID], nodeID % 2 == 1, getDim(buffer[parentID]), dim,
                              exp_x_lu /*+ offset*/, exp_y_lu /*+ offset*/, exp_z_lu /*+ offset*/) +
                          refPos;

                    /*pos = getParticle(buffer[particleID], nodeID % 2 == 1,
                        #ifdef E4M16
                        buffer[parentID].dim,
                        #else
                              (buffer[parentID].dim_b << 1) + buffer[parentID].dim_a,
                        #endif
                                          dim, exp_x_lu + offset, exp_y_lu + offset, exp_z_lu + offset) +
                                      refPos;*/
                }

                //getParticle(treelet, self.qparticleBuffer[particleID], dim, pos);
                /*const pkd::Particle particle = self.particleBuffer[particleID];
                int const dim = particle.dim;*/
                //pos = pos * refPos;

                const float t_slab_lo = (pos[dim] - self.particleRadius - org[dim]) * rdir[dim];
                const float t_slab_hi = (pos[dim] + self.particleRadius - org[dim]) * rdir[dim];

                const float t_slab_nr = fminf(t_slab_lo, t_slab_hi);
                const float t_slab_fr = fmaxf(t_slab_lo, t_slab_hi);

                // -------------------------------------------------------
                // compute potential sphere interval, and intersect if necessary
                // -------------------------------------------------------
                const float sphere_t0 = fmaxf(t0, t_slab_nr);
                const float sphere_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);

                if (sphere_t0 < sphere_t1) {
                    if (intersectSphere(pos, self.particleRadius, ray, tmp_hit_t)) {
                        tmp_hit_primID = particleID;

                        tmp_hit_pos = pos;
                    }
                }

                // -------------------------------------------------------
                // compute near and far side intervals
                // -------------------------------------------------------
                const float nearSide_t0 = t0;
                const float nearSide_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);

                const float farSide_t0 = fmaxf(t0, t_slab_nr);
                const float farSide_t1 = fminf(t1, tmp_hit_t);

                // -------------------------------------------------------
                // logic
                // -------------------------------------------------------
                const int nearSide_nodeID = 2 * nodeID + 1 + dir_sign[dim];
                const int farSide_nodeID = 2 * nodeID + 2 - dir_sign[dim];

                const bool nearSide_valid = nearSide_nodeID < size;
                const bool farSide_valid = farSide_nodeID < size;

                const bool need_nearSide = nearSide_valid && nearSide_t0 < nearSide_t1;
                const bool need_farSide = farSide_valid && farSide_t0 < farSide_t1;

                if (!(need_nearSide || need_farSide))
                    break; // pop ...

                if (need_nearSide && need_farSide) {
                    stackPtr->nodeID = farSide_nodeID;
                    stackPtr->t0 = farSide_t0;
                    stackPtr->t1 = farSide_t1;
                    //setEntry(stackPtr, farSide_nodeID, farSide_t0, farSide_t1);

                    stackPtr->refPos = pos;

                    ++stackPtr;

                    nodeID = nearSide_nodeID;
                    t0 = nearSide_t0;
                    t1 = nearSide_t1;

                    refPos = pos;


                    continue;
                }

                nodeID = need_nearSide ? nearSide_nodeID : farSide_nodeID;
                t0 = need_nearSide ? nearSide_t0 : farSide_t0;
                t1 = need_nearSide ? nearSide_t1 : farSide_t1;

                refPos = pos;
            }
            // -------------------------------------------------------
            // pop
            // -------------------------------------------------------
            while (1) {
                if (stackPtr == stackBase) {
                    // can't pop any more - done.
                    if (tmp_hit_primID >= 0 && tmp_hit_t < ray.tmax) {
                        prd.pos = tmp_hit_pos;
                        optixReportIntersection(tmp_hit_t, 0, tmp_hit_primID);
                    }
                    return;
                }
                --stackPtr;
                //getEntry(stackPtr, nodeID, t0, t1);
                t0 = stackPtr->t0;
                t1 = stackPtr->t1;
                nodeID = stackPtr->nodeID;
                t1 = fminf(t1, tmp_hit_t);

                refPos = stackPtr->refPos;

                if (t1 <= t0)
                    continue;
                break;
            }
        }
    }
}

template<typename BUF_TYPE, int BEXP = BUF_TYPE::exp>
void __device__ traverse(BUF_TYPE const* buffer) {
    const int treeletID = optixGetPrimitiveIndex();
    const auto& self = owl::getProgramData<FloatCompGeomData>();
    const auto& treelet = self.treeletBuffer[treeletID];
    PerRayData& prd = owl::getPRD<PerRayData>();
    //auto const localTables = self.localTables;

    int const num_idx = powf(2, BEXP);

    char const* exp_x_lu = self.expXBuffer;
    char const* exp_y_lu = self.expYBuffer;
    char const* exp_z_lu = self.expZBuffer;

    if (self.use_localtables > 0) {
        exp_x_lu += treeletID * num_idx;
        exp_y_lu += treeletID * num_idx;
        exp_z_lu += treeletID * num_idx;
    }

    auto const ray = owl::Ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

    const int begin = treelet.begin;
    const int size = treelet.end - begin;
    {
        float t0, t1;
        if (!clipToBounds(ray, treelet.bounds, t0, t1))
            return;


        int nodeID = 0;
        float tmp_hit_t = t1;
        int tmp_hit_primID = -1;

        enum { STACK_DEPTH = 12 };

        FloatCompStackEntry stackBase[STACK_DEPTH];
        FloatCompStackEntry* stackPtr = stackBase;

        const int dir_sign[3] = {ray.direction.x < 0.f, ray.direction.y < 0.f, ray.direction.z < 0.f};
        const float org[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
        const float rdir[3] = {
            (fabsf(ray.direction.x) <= 1e-8f) ? 1e8f : 1.f / ray.direction.x,
            (fabsf(ray.direction.y) <= 1e-8f) ? 1e8f : 1.f / ray.direction.y,
            (fabsf(ray.direction.z) <= 1e-8f) ? 1e8f : 1.f / ray.direction.z,
        };


        vec3f pos;
        int dim;
        vec3f refPos = treelet.basePos;

        vec3f tmp_hit_pos;


        //const float radius = self.radius;

        //unsigned int baseOffset = 0;
        /*if (localTables) {
            baseOffset = 1;
        }*/

        //auto const* buffer = (QTParticle_e4m16 const*)self.qtpBuffer;
        while (1) {
            // while we have anything to traverse ...

            while (1) {
                // while we can go down

                const int particleID = nodeID + begin;

                //auto test = self.qtpBuffer[particleID];

                //getParticle(treelet, self.qtpBuffer[particleID], dim, pos, exp_x_lu, exp_y_lu, exp_z_lu);

                //unsigned int offset = treeletID * num_idx * baseOffset;
                pos = getParticle(buffer[particleID], dim, exp_x_lu /*+ offset*/, exp_y_lu /*+ offset*/,
                          exp_z_lu /*+ offset*/) +
                      refPos;


                //getParticle(treelet, self.qparticleBuffer[particleID], dim, pos);
                /*const pkd::Particle particle = self.particleBuffer[particleID];
                int const dim = particle.dim;*/
                //pos = pos * refPos;

                const float t_slab_lo = (pos[dim] - self.particleRadius - org[dim]) * rdir[dim];
                const float t_slab_hi = (pos[dim] + self.particleRadius - org[dim]) * rdir[dim];

                const float t_slab_nr = fminf(t_slab_lo, t_slab_hi);
                const float t_slab_fr = fmaxf(t_slab_lo, t_slab_hi);

                // -------------------------------------------------------
                // compute potential sphere interval, and intersect if necessary
                // -------------------------------------------------------
                const float sphere_t0 = fmaxf(t0, t_slab_nr);
                const float sphere_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);

                if (sphere_t0 < sphere_t1) {
                    if (intersectSphere(pos, self.particleRadius, ray, tmp_hit_t)) {
                        tmp_hit_primID = particleID;
                        tmp_hit_pos = pos;
                    }
                }

                // -------------------------------------------------------
                // compute near and far side intervals
                // -------------------------------------------------------
                const float nearSide_t0 = t0;
                const float nearSide_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);

                const float farSide_t0 = fmaxf(t0, t_slab_nr);
                const float farSide_t1 = fminf(t1, tmp_hit_t);

                // -------------------------------------------------------
                // logic
                // -------------------------------------------------------
                const int nearSide_nodeID = 2 * nodeID + 1 + (ray.direction[dim] < 0.f);
                const int farSide_nodeID = 2 * nodeID + 2 - (ray.direction[dim] < 0.f);

                const bool nearSide_valid = nearSide_nodeID < size;
                const bool farSide_valid = farSide_nodeID < size;

                const bool need_nearSide = nearSide_valid && nearSide_t0 < nearSide_t1;
                const bool need_farSide = farSide_valid && farSide_t0 < farSide_t1;

                if (!(need_nearSide || need_farSide))
                    break; // pop ...

                if (need_nearSide && need_farSide) {
                    stackPtr->nodeID = farSide_nodeID;
                    stackPtr->t0 = farSide_t0;
                    stackPtr->t1 = farSide_t1;
                    //setEntry(stackPtr, farSide_nodeID, farSide_t0, farSide_t1);

                    ++stackPtr;

                    nodeID = nearSide_nodeID;
                    t0 = nearSide_t0;
                    t1 = nearSide_t1;


                    continue;
                }

                nodeID = need_nearSide ? nearSide_nodeID : farSide_nodeID;
                t0 = need_nearSide ? nearSide_t0 : farSide_t0;
                t1 = need_nearSide ? nearSide_t1 : farSide_t1;
            }
            // -------------------------------------------------------
            // pop
            // -------------------------------------------------------
            while (1) {
                if (stackPtr == stackBase) {
                    // can't pop any more - done.
                    if (tmp_hit_primID >= 0 && tmp_hit_t < ray.tmax) {
                        prd.pos = tmp_hit_pos;
                        optixReportIntersection(tmp_hit_t, 0, tmp_hit_primID);
                    }
                    return;
                }
                --stackPtr;
                //getEntry(stackPtr, nodeID, t0, t1);
                t0 = stackPtr->t0;
                t1 = stackPtr->t1;
                nodeID = stackPtr->nodeID;
                t1 = fminf(t1, tmp_hit_t);

                if (t1 <= t0)
                    continue;
                break;
            }
        }
    }
}

OPTIX_INTERSECT_PROGRAM(floatcomp_intersect_e5m15d)() {
    const auto& self = owl::getProgramData<FloatCompGeomData>();
    auto const* buffer = (QTParticle_e5m15d const*) self.particleBuffer;
    traverse_dep(buffer);
}

OPTIX_INTERSECT_PROGRAM(floatcomp_intersect_e5m15)() {
    const auto& self = owl::getProgramData<FloatCompGeomData>();
    auto const* buffer = (QTParticle_e5m15 const*) self.particleBuffer;
    traverse(buffer);
}

OPTIX_CLOSEST_HIT_PROGRAM(floatcomp_ch)() {
    PerRayData& prd = owl::getPRD<PerRayData>();
    const auto& self = owl::getProgramData<FloatCompGeomData>();
    prd.particleID = optixGetAttribute_0();
    prd.t = optixGetRayTmax();
}

OPTIX_BOUNDS_PROGRAM(floatcomp_bounds)(const void* geomData, box3f& primBounds, const int primID) {
    primBounds = ((const FloatCompGeomData*) geomData)->treeletBuffer[primID].bounds;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
