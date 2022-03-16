/*
 * Modified by MegaMol Dev Team
 * Based on project "ospray-module-pkd" files "ParticleModel.h" and "ParticleModel.cpp" (Apache License 2.0)
 */

#include "pkd/ParticleModel.h"
#include "stdafx.h"

#include "mmcore/utility/log/Log.h"

using namespace megamol;


VISLIB_FORCEINLINE float floatFromVoidArray(const geocalls::MultiParticleDataCall::Particles& p, size_t index) {
    // const float* parts = static_cast<const float*>(p.GetVertexData());
    // return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}


VISLIB_FORCEINLINE unsigned char byteFromVoidArray(const geocalls::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetVertexData())[index];
}


VISLIB_FORCEINLINE float floatColFromVoidArray(const geocalls::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const float*>(p.GetColourData())[index];
}


VISLIB_FORCEINLINE unsigned char byteColFromVoidArray(
    const geocalls::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetColourData())[index];
}


typedef float (*floatFromArrayFunc)(const geocalls::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char (*byteFromArrayFunc)(const geocalls::MultiParticleDataCall::Particles& p, size_t index);

typedef float (*floatColFromArrayFunc)(const geocalls::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char (*byteColFromArrayFunc)(const geocalls::MultiParticleDataCall::Particles& p, size_t index);


inline rkcommon::math::vec3f makeRandomColor(const int i) {
    const int mx = 13 * 17 * 43;
    const int my = 11 * 29;
    const int mz = 7 * 23 * 63;
    const uint32_t g = (i * (3 * 5 * 127) + 12312314);
    return rkcommon::math::vec3f((g % mx) * (1.f / (mx - 1)), (g % my) * (1.f / (my - 1)), (g % mz) * (1.f / (mz - 1)));
}


//! return world bounding box of all particle *positions* (i.e., particles *ex* radius)
rkcommon::math::box3f ospray::ParticleModel::getBounds() const {
    rkcommon::math::box3f bounds = rkcommon::math::empty;
    for (size_t i = 0; i < position.size(); ++i)
        bounds.extend({position[i].x, position[i].y, position[i].z});
    return bounds;
}


void megamol::ospray::ParticleModel::fill(geocalls::SimpleSphericalParticles parts) {
    // Attribute rgba("rgba");

    size_t vertexLength;
    size_t colorLength;

    // Vertex data type check
    if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
        vertexLength = 3;
    } else if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        vertexLength = 4;
    }

    auto const& parStore = parts.GetParticleStore();
    auto const& xAcc = parStore.GetXAcc();
    auto const& yAcc = parStore.GetYAcc();
    auto const& zAcc = parStore.GetZAcc();
    auto const& rAcc = parStore.GetCRAcc();
    auto const& gAcc = parStore.GetCGAcc();
    auto const& bAcc = parStore.GetCBAcc();
    auto const& aAcc = parStore.GetCAAcc();

    for (size_t loop = 0; loop < parts.GetCount(); ++loop) {

        rkcommon::math::vec3f pos;

        pos.x = xAcc->Get_f(loop);
        pos.y = yAcc->Get_f(loop);
        pos.z = zAcc->Get_f(loop);

        rkcommon::math::vec4uc col;

        col.x = rAcc->Get_u8(loop);
        col.y = gAcc->Get_u8(loop);
        col.z = bAcc->Get_u8(loop);
        col.w = aAcc->Get_u8(loop);

        float const color = encodeColorToFloat(col);

        this->position.emplace_back(pos, color);
    }
}
