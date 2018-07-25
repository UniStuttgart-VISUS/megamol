#include "stdafx.h"
#include "pkd/MMPKDParticleModel.h"

#include "vislib/sys/Log.h"

using namespace megamol;

void megamol::ospray::MMPKDParticleModel::fill(megamol::core::moldyn::SimpleSphericalParticles const& parts) {
    auto const vtype = parts.GetVertexDataType();
    auto const ctype = parts.GetColourDataType();

    if (vtype == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR ||
        vtype == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ) {
        throw std::runtime_error("MMPKDParticleModel does not support FLOAT_XYZR or SHORT_XYZ");
    }

    auto mmbbox = parts.GetBBox();
    bbox.lower.x = mmbbox.GetLeft();
    bbox.lower.y = mmbbox.GetBottom();
    bbox.lower.z = mmbbox.GetBack();
    bbox.upper.x = mmbbox.GetRight();
    bbox.upper.y = mmbbox.GetTop();
    bbox.upper.z = mmbbox.GetFront();

    if (vtype == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
        if (ctype != megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
            vislib::sys::Log::DefaultLog.WriteWarn("MMPKDParticleModel only supports UINT8_RGBA for FLOAT_XYZ, falling back to UINT8_RGBA");
        }

        doublePrecision = false;

        // in this case use positionf
        auto const pcount = parts.GetCount();
        for (size_t pidx = 0; pidx < pcount; ++pidx) {
            auto part = parts[pidx];

            ospcommon::vec3f pos;
            pos.x = part.vert.GetXf();
            pos.y = part.vert.GetYf();
            pos.z = part.vert.GetZf();

            ospcommon::vec4uc col;
            col.x = part.col.GetRu8();
            col.y = part.col.GetGu8();
            col.z = part.col.GetBu8();
            col.w = part.col.GetAu8();

            float color = 0.0f;
            auto const ptr = reinterpret_cast<unsigned char*>(&color);
            ptr[0] = col.x;
            ptr[1] = col.y;
            ptr[2] = col.z;
            ptr[3] = col.w;

            this->positionf.emplace_back(pos, color);
        }

        numParticles = this->positionf.size();
    } else if (vtype == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ) {
        if (ctype != megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA) {
            vislib::sys::Log::DefaultLog.WriteWarn("MMPKDParticleModel only supports SHORT_RGBA for FLOAT_XYZ, falling back to SHORT_RGBA");
        }

        doublePrecision = true;

        // in this case use positiond
        auto const pcount = parts.GetCount();
        for (size_t pidx = 0; pidx < pcount; ++pidx) {
            auto part = parts[pidx];

            ospcommon::vec3d pos;
            pos.x = part.vert.GetXf();
            pos.y = part.vert.GetYf();
            pos.z = part.vert.GetZf();

            ospcommon::vec4uc col;
            col.x = part.col.GetRu8();
            col.y = part.col.GetGu8();
            col.z = part.col.GetBu8();
            col.w = part.col.GetAu8();

            double color = 0.0;
            auto const ptr = reinterpret_cast<unsigned short*>(&color);
            ptr[0] = static_cast<unsigned short>(col.x) * 257;
            ptr[1] = static_cast<unsigned short>(col.y) * 257;
            ptr[2] = static_cast<unsigned short>(col.z) * 257;
            ptr[3] = static_cast<unsigned short>(col.w) * 257;

            this->positiond.emplace_back(pos, color);
        }

        numParticles = this->positiond.size();
    }
}
