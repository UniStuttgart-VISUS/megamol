/*
 * ParticleVisibilityFromVolume.h
 *
 * Copyright (C) 2018 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleVisibilityFromVolume.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleVisibilityFromVolume::ParticleVisibilityFromVolume
 */
datatools::ParticleVisibilityFromVolume::ParticleVisibilityFromVolume(void)
    : AbstractParticleManipulator("outData", "indata")
    , operatorSlot("operator", "what to do with the reference value")
    , valueSlot("ref", "the value for the operator")
    , epsilonSlot("epsilon", "the tolerance for equality")
    , absoluteSlot("absolute", "use absolute values instead of relative")
    //, cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction")
    //, cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction")
    //, cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction")
    , minSlot("min", "the minimum value in the volume (read only)")
    , maxSlot("max", "the maximum value in the volume (read only)")
    , volumeSlot("volume", "the volume we the operation is based on") {

    this->valueSlot.SetParameter(new core::param::FloatParam(0.5, -5000.0f, 5000.0f));
    this->MakeSlotAvailable(&this->valueSlot);
    this->epsilonSlot.SetParameter(new core::param::FloatParam(0.001, 0.0f, 5000.0f));
    this->MakeSlotAvailable(&this->epsilonSlot);
    this->absoluteSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->absoluteSlot);
    this->minSlot.SetParameter(new core::param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minSlot);
    this->maxSlot.SetParameter(new core::param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxSlot);

    //this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    //this->MakeSlotAvailable(&this->cyclXSlot);
    //this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    //this->MakeSlotAvailable(&this->cyclYSlot);
    //this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    //this->MakeSlotAvailable(&this->cyclZSlot);

    auto ep = new megamol::core::param::EnumParam(0);
    ep->SetTypePair(0, "smaller");
    ep->SetTypePair(1, "larger");
    ep->SetTypePair(2, "equal");
    this->operatorSlot << ep;
    this->MakeSlotAvailable(&this->operatorSlot);

    this->volumeSlot.SetCompatibleCall<megamol::core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->volumeSlot);
}


/*
 * datatools::ParticleVisibilityFromVolume::~ParticleVisibilityFromVolume
 */
datatools::ParticleVisibilityFromVolume::~ParticleVisibilityFromVolume(void) { this->Release(); }


/*
 * datatools::ParticleThinner::manipulateData
 */
bool datatools::ParticleVisibilityFromVolume::manipulateData(
    megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::misc::VolumetricDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    float theVal = this->valueSlot.Param<core::param::FloatParam>()->Value();
    float epsilon = this->epsilonSlot.Param<core::param::FloatParam>()->Value();
    auto op = this->operatorSlot.Param<core::param::EnumParam>()->Value();
    auto absolute = this->absoluteSlot.Param<core::param::BoolParam>()->Value();

    auto* inVol = this->volumeSlot.CallAs<VolumetricDataCall>();
    inVol->SetFrameID(outData.FrameID());
    if (!(*inVol)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleVisibilityFromVolume: cannot get extents of volume");
        return false;
    }
    if (!(*inVol)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleVisibilityFromVolume: cannot get volume data");
        return false;
    }
    if (inVol->FrameID() != inData.FrameID()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "ParticleVisibilityFromVolume: frameIDs of particles and volume do not match: %u (vol) - %u (parts)", inVol->FrameID(),
            inData.FrameID());
        return false;
    }

    if (!inVol->IsUniform(0) || !inVol->IsUniform(1) || !inVol->IsUniform(2)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleVisibilityFromVolume: input Volume has to be uniform!");
        return false;
    }

    if (inData.FrameID() == this->lastTime && inData.DataHash() == this->lastParticleHash &&
        inVol->DataHash() == this->lastVolumeHash && !operatorSlot.IsDirty() && !valueSlot.IsDirty() && !epsilonSlot.IsDirty()
        && !absoluteSlot.IsDirty()) {
        // everything should already be correct
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteInfo("Computing Visibility from Volume...");

    const VolumetricDataCall::Metadata* volMeta = inVol->GetMetadata();
    const auto theVol = inVol->GetData();

    if (!volMeta->GridType == VolumetricDataCall::GridType::CARTESIAN ||
        !volMeta->GridType == VolumetricDataCall::GridType::RECTILINEAR) {
        vislib::sys::Log::DefaultLog.WriteError(
            "ParticleVisibilityFromVolume: input Volume has to be cartesian or rectilinear!");
        return false;
    }

    const uint32_t channel = 0;

    this->minSlot.Param<core::param::FloatParam>()->SetValue(volMeta->MinValues[channel]);
    this->maxSlot.Param<core::param::FloatParam>()->SetValue(volMeta->MaxValues[channel]);

    bool volumeIsNotBBoxAligned = false;

    //bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
    //bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
    //bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();

    typedef const float (VolumetricDataCall::*getter)(const uint32_t, const uint32_t, const uint32_t, const uint32_t) const;
    getter getFun;
    if (absolute) {
        getFun = &VolumetricDataCall::GetAbsoluteVoxelValue;
    } else {
        getFun = &VolumetricDataCall::GetRelativeVoxelValue;
    }

    vislib::sys::Log::DefaultLog.WriteInfo("ParticleVisibilityFromVolume: starting filtering");
    const auto startTime = std::chrono::high_resolution_clock::now();

    unsigned int plc = inData.GetParticleListCount();
    this->theVertexData.resize(plc);
    this->theColorData.resize(plc);
    outData.SetParticleListCount(plc);
    for (unsigned int i = 0; i < plc; ++i) {
        MultiParticleDataCall::Particles& p = inData.AccessParticles(i);

        const UINT64 cnt = p.GetCount();
        const void* cd = p.GetColourData();
        const unsigned int cdstride = p.GetColourDataStride();
        const MultiParticleDataCall::Particles::ColourDataType cdt = p.GetColourDataType();
        const unsigned int cdsize = MultiParticleDataCall::Particles::ColorDataSize[cdt];

        const void* vd = p.GetVertexData();
        unsigned int vdstride = p.GetVertexDataStride();
        const MultiParticleDataCall::Particles::VertexDataType vdt = p.GetVertexDataType();
        const auto vdsize = MultiParticleDataCall::Particles::VertexDataSize[vdt];

        UINT64 cntLeft = 0;

        const uint8_t* commonBasePointer = nullptr;
        const uint8_t* vertexBasePointer = nullptr;
        const uint8_t* colorBasePointer = nullptr;
        bool isInterleaved = false;
        bool colorIsFirst = false;

        vertexBasePointer = reinterpret_cast<const uint8_t*>(p.GetVertexData());
        colorBasePointer = reinterpret_cast<const uint8_t*>(p.GetColourData());

        if (cdsize == 0 || vdsize + cdsize == vdstride) {
            // data is interleaved
            commonBasePointer = vertexBasePointer < colorBasePointer ? vertexBasePointer : colorBasePointer;
            colorIsFirst = vertexBasePointer > colorBasePointer;
            isInterleaved = true;
            vdstride = vdsize + cdsize;
            theVertexData[i].resize(cnt * vdstride);
        } else {
            theVertexData[i].resize(cnt * vdsize);
            theColorData[i].resize(cnt * cdsize);
        }

#ifdef _OPENMP
        const auto numThreads = omp_get_num_threads();
#endif
        // todo: is this OK?
        #pragma omp parallel for
        for (INT64 j = 0; j < cnt; ++j) {
            const auto x = p[j].vert.GetXf();
            const auto y = p[j].vert.GetYf();
            const auto z = p[j].vert.GetZf();

            // relative coordinates in volume
            const auto rx = (x - volMeta->Origin[0]) / volMeta->SliceDists[0][0];
            const auto ry = (y - volMeta->Origin[1]) / volMeta->SliceDists[1][0];
            const auto rz = (z - volMeta->Origin[2]) / volMeta->SliceDists[2][0];

            const int quantX = static_cast<int>(rx);
            const int quantY = static_cast<int>(ry);
            const int quantZ = static_cast<int>(rz);
            int quantX2 = quantX;
            int quantY2 = quantY;
            int quantZ2 = quantZ;
            const float diffX = rx - quantX;
            const float diffY = ry - quantY;
            const float diffZ = rz - quantZ;

            if (quantX >= 0 || quantX < static_cast<int>(volMeta->Resolution[0]) || quantY >= 0 ||
                quantY < static_cast<int>(volMeta->Resolution[1]) || quantZ >= 0 ||
                quantZ < static_cast<int>(volMeta->Resolution[2])) {
                // OK actually
            } else {
                volumeIsNotBBoxAligned = true;
                continue;
            }

            // warning: we fake CBC here which has no relevance since at the right border,
            // the right neighbor influence should be pulled to 0 anyway
            quantX2 = std::max<size_t>(0, std::min<size_t>(volMeta->Resolution[0] - 1, quantX + 1));
            quantY2 = std::max<size_t>(0, std::min<size_t>(volMeta->Resolution[1] - 1, quantY + 1));
            quantZ2 = std::max<size_t>(0, std::min<size_t>(volMeta->Resolution[2] - 1, quantZ + 1));

            const float c000 = ((inVol)->*(getFun))(quantX, quantY, quantZ, channel);
            const float c100 = ((inVol)->*(getFun))(quantX2, quantY, quantZ, channel);
            const float c010 = ((inVol)->*(getFun))(quantX, quantY2, quantZ, channel);
            const float c110 = ((inVol)->*(getFun))(quantX2, quantY2, quantZ, channel);
            const float c001 = ((inVol)->*(getFun))(quantX, quantY, quantZ2, channel);
            const float c101 = ((inVol)->*(getFun))(quantX2, quantY, quantZ2, channel);
            const float c011 = ((inVol)->*(getFun))(quantX, quantY2, quantZ2, channel);
            const float c111 = ((inVol)->*(getFun))(quantX2, quantY2, quantZ2, channel);

            float volVal = (1.0f - diffX) * (1.0f - diffY) * (1.0f - diffZ) * c000 +
                           diffX * (1.0f - diffY) * (1.0f - diffZ) * c100 +
                           (1.0f - diffX) * diffY * (1.0f - diffZ) * c010 + diffX * diffY * (1.0f - diffZ) * c110 +
                           (1.0f - diffX) * (1.0f - diffY) * diffZ * c001 + diffX * (1.0f - diffY) * diffZ * c101 +
                           (1.0f - diffX) * diffY * diffZ * c011 + diffX * diffY * diffZ * c111;

            bool isOK = false;
            switch (op) {
            case 0:
                // smaller
                isOK = volVal < theVal;
                break;
            case 1:
                // larger
                isOK = volVal > theVal;
                break;
            case 2:
                // equal
                isOK = std::abs(volVal - theVal) < epsilon;
                break;
            }

            if (isOK) {
#ifdef _OPENMP
                const UINT64 localIdx = cntLeft + omp_get_thread_num();
#else
                const UINT64 localIdx = cntLeft;
#endif
                if (isInterleaved) {
                    memcpy(theVertexData[i].data() + vdstride * localIdx, commonBasePointer + vdstride * j, vdstride);
                } else {
                    memcpy(theVertexData[i].data() + vdsize * localIdx, vertexBasePointer + vdstride * j, vdsize);
                    memcpy(theColorData[i].data() + cdsize * localIdx, colorBasePointer + cdstride * j, cdsize);
                }
                #pragma omp atomic
                cntLeft++;
            }
        }

        auto& outp = outData.AccessParticles(i);
        outp.SetCount(cntLeft);
        vislib::sys::Log::DefaultLog.WriteInfo(
            "ParticleVisibilityFromVolume: list %d: %lu / %lu particles left", i, cntLeft, cnt);
        if (isInterleaved) {
            theVertexData[i].resize(cntLeft * vdstride);
            //theColorData[i].resize(cntLeft * cdsize);
        } else {
            theVertexData[i].resize(cntLeft * vdsize);
            theColorData[i].resize(cntLeft * cdsize);
        }
        auto col = p.GetGlobalColour();
        outp.SetGlobalColour(col[0], col[1], col[2], col[3]);
        outp.SetGlobalRadius(p.GetGlobalRadius());
        outp.SetGlobalType(p.GetGlobalType());
        if (isInterleaved) {
            if (colorIsFirst) {
                outp.SetColourData(cdt, theVertexData[i].data(), cdstride);
                outp.SetVertexData(vdt, theVertexData[i].data() + (vertexBasePointer - colorBasePointer), vdstride);
            } else {
                outp.SetVertexData(vdt, theVertexData[i].data(), vdstride);
                outp.SetColourData(cdt, theVertexData[i].data() + (colorBasePointer - vertexBasePointer), cdstride);
            }
        } else {
            outp.SetVertexData(vdt, theVertexData[i].data(), vdsize);
            outp.SetColourData(cdt, theColorData[i].data(), cdsize);
        }

    }

    const auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diffMillis = endTime - startTime;
    vislib::sys::Log::DefaultLog.WriteInfo("ParticleVisibilityFromVolume took %f ms.", diffMillis.count());


    if (volumeIsNotBBoxAligned) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "ParticleVisibilityFromVolume: Volume does not cover all of the domain!");
    }

    this->lastTime = inData.FrameID();
    this->lastParticleHash = inData.DataHash();
    this->lastVolumeHash = inVol->DataHash();

    this->operatorSlot.ResetDirty();
    this->valueSlot.ResetDirty();
    this->epsilonSlot.ResetDirty();
    this->absoluteSlot.ResetDirty();

    return true;
}
