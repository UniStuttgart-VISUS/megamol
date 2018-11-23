#include "stdafx.h"
#include "HMMPKDParticleModel.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "vislib/sys/Log.h"


void HMMPKDParticleModel::fill(megamol::core::moldyn::SimpleSphericalParticles const& parts) {
    auto const vtype = parts.GetVertexDataType();
    auto const ctype = parts.GetColourDataType();

    this->position.clear();
    this->pkd_particle.clear();

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


    if (ctype != megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "HMMPKDParticleModel only supports SHORT_RGBA for DOUBLE_XYZ, falling back to SHORT_RGBA");
    }

    auto const pcount = parts.GetCount();
    for (size_t pidx = 0; pidx < pcount; ++pidx) {
        auto part = parts[pidx];

        ospcommon::vec3d pos;
        pos.x = part.vert.GetXd();
        pos.y = part.vert.GetYd();
        pos.z = part.vert.GetZd();

        ospcommon::vec4ui col;
        if (!globCol) {
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
        ptr[0] = static_cast<unsigned short>(col.x); // * 257;
        ptr[1] = static_cast<unsigned short>(col.y); // * 257;
        ptr[2] = static_cast<unsigned short>(col.z); // * 257;
        ptr[3] = static_cast<unsigned short>(col.w); // * 257;

        this->position.emplace_back(pos, color);
    }

    numParticles = this->position.size();
    this->pkd_particle.resize(numParticles);

    radius = parts.GetGlobalRadius();
}

