/*
 * Modified by MegaMol Dev Team
 * Based on project "ospray-module-pkd" files "ParticleModel.h" and "ParticleModel.cpp" (Apache License 2.0)
 */

#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "geometry_calls/MultiParticleDataCall.h"

#include "rkcommon/math//box.h"
#include "rkcommon/math/vec.h"


namespace megamol {
namespace ospray {

/*! complete input data for a particle model */
struct ParticleModel {

    ParticleModel() : radius(0) {}

    std::vector<rkcommon::math::vec4f> position; //!< particle position + color encoded in 'w'

    void fill(geocalls::SimpleSphericalParticles parts);

    //! return world bounding box of all particle *positions* (i.e., particles *ex* radius)
    rkcommon::math::box3f getBounds() const;

    float encodeColorToFloat(rkcommon::math::vec4f const& col) {
        unsigned int r = static_cast<unsigned int>(col.x * 255.f);
        unsigned int g = static_cast<unsigned int>(col.y * 255.f);
        unsigned int b = static_cast<unsigned int>(col.z * 255.f);
        unsigned int a = static_cast<unsigned int>(col.w * 255.f);

        unsigned int color = (r << 24) + (g << 16) + (b << 8) + a;

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float encodeColorToFloat(rkcommon::math::vec4uc const& col) {
        unsigned int r = static_cast<unsigned int>(col.x);
        unsigned int g = static_cast<unsigned int>(col.y);
        unsigned int b = static_cast<unsigned int>(col.z);
        unsigned int a = static_cast<unsigned int>(col.w);

        unsigned int color = (r << 24) + (g << 16) + (b << 8) + a;

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float encodeColorToFloat(rkcommon::math::vec3f const& col) {
        unsigned int r = static_cast<unsigned int>(col.x * 255.f);
        unsigned int g = static_cast<unsigned int>(col.y * 255.f);
        unsigned int b = static_cast<unsigned int>(col.z * 255.f);

        unsigned int color = (r << 24) + (g << 16) + (b << 8);

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float encodeColorToFloat(rkcommon::math::vec3uc const& col) {
        unsigned int r = static_cast<unsigned int>(col.x);
        unsigned int g = static_cast<unsigned int>(col.y);
        unsigned int b = static_cast<unsigned int>(col.z);

        unsigned int color = (r << 24) + (g << 16) + (b << 8);

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float radius; //!< radius to use (0 if not specified)
};

} /* end namespace ospray */
} /* end namespace megamol */
