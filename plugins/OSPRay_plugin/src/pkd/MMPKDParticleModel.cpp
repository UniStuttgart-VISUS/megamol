#include "stdafx.h"
#include "pkd/MMPKDParticleModel.h"

#include "vislib/sys/Log.h"

using namespace megamol;

void megamol::ospray::MMPKDParticleModel::fill(megamol::core::moldyn::SimpleSphericalParticles const& parts) {
    auto const vtype = parts.GetVertexDataType();
    auto const ctype = parts.GetColourDataType();

    this->positionf.clear();
    this->positiond.clear();

    if (vtype == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR ||
        vtype == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ) {
        throw std::runtime_error("MMPKDParticleModel does not support FLOAT_XYZR or SHORT_XYZ");
    }

    bool globCol = false;
    uint8_t globalColor_u8[] = {0, 0, 0, 0};
    uint16_t globalColor_u16[] = {0, 0, 0, 0};
    auto gc = parts.GetGlobalColour();
    if (ctype == megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        globalColor_u8[0] = gc[0];
        globalColor_u8[1] = gc[1];
        globalColor_u8[2] = gc[2];
        globalColor_u8[3] = gc[3];
        globalColor_u16[0] = gc[0] * 257;
        globalColor_u16[1] = gc[1] * 257;
        globalColor_u16[2] = gc[2] * 257;
        globalColor_u16[3] = gc[3] * 257;
        globCol = true;
    }

    auto mmbbox = parts.GetBBox();
    lbbox.lower.x = mmbbox.GetLeft();
    lbbox.lower.y = mmbbox.GetBottom();
    lbbox.lower.z = mmbbox.GetBack();
    lbbox.upper.x = mmbbox.GetRight();
    lbbox.upper.y = mmbbox.GetTop();
    lbbox.upper.z = mmbbox.GetFront();

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
            if (!globCol) {
                col.x = part.col.GetRu8();
                col.y = part.col.GetGu8();
                col.z = part.col.GetBu8();
                col.w = part.col.GetAu8();
            } else {
                col.x = globalColor_u8[0];
                col.y = globalColor_u8[1];
                col.z = globalColor_u8[2];
                col.w = globalColor_u8[3];
            }

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
            pos.x = part.vert.GetXd();
            pos.y = part.vert.GetYd();
            pos.z = part.vert.GetZd();

            ospcommon::vec4ui col;
            if (globCol) {
                col.x = part.col.GetRu16();
                col.y = part.col.GetGu16();
                col.z = part.col.GetBu16();
                col.w = part.col.GetAu16();
            } else {
                col.x = globalColor_u16[0];
                col.y = globalColor_u16[1];
                col.z = globalColor_u16[2];
                col.w = globalColor_u16[3];
            }

            double color = 0.0;
            auto const ptr = reinterpret_cast<unsigned short*>(&color);
            ptr[0] = static_cast<unsigned short>(col.x);// * 257;
            ptr[1] = static_cast<unsigned short>(col.y);// * 257;
            ptr[2] = static_cast<unsigned short>(col.z);// * 257;
            ptr[3] = static_cast<unsigned short>(col.w);// * 257;

            this->positiond.emplace_back(pos, color);
        }

        numParticles = this->positiond.size();
    }

    radius = parts.GetGlobalRadius();
}
