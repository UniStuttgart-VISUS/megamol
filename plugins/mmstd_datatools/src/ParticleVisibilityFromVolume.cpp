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
    , volumeSlot("volume", "the volume we the operation is based on") {

    this->valueSlot.SetParameter(new core::param::FloatParam(0.5, -5000.0f, 5000.0f));
    this->MakeSlotAvailable(&this->valueSlot);
    this->epsilonSlot.SetParameter(new core::param::FloatParam(0.001, 0.0f, 5000.0f));
    this->MakeSlotAvailable(&this->epsilonSlot);

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
            "ParticleVisibilityFromVolume: frameIDs of particles and volume do not match");
        return false;
    }

    if (!inVol->IsUniform(0) || !inVol->IsUniform(1) || !inVol->IsUniform(2)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleVisibilityFromVolume: input Volume has to be uniform!");
        return false;
    }

    if (inData.FrameID() == this->lastTime && inData.DataHash() == this->lastParticleHash &&
        inVol->DataHash() == this->lastVolumeHash && !operatorSlot.IsDirty() && !valueSlot.IsDirty() && !epsilonSlot.IsDirty()) {
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
        const unsigned int vdstride = p.GetVertexDataStride();
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
            theVertexData[i].reserve(cnt * vdstride);
        } else {
            theVertexData[i].reserve(cnt * vdsize);
            theColorData[i].reserve(cnt * cdsize);
        }

        for (INT64 j = 0; j < cnt; ++j) {
            const auto x = p[j].vert.GetXf();
            const auto y = p[j].vert.GetYf();
            const auto z = p[j].vert.GetZf();

            // relative coordinates in volume
            const auto rx = (x - volMeta->Origin[0]) / volMeta->SliceDists[0][0];
            const auto ry = (x - volMeta->Origin[1]) / volMeta->SliceDists[1][0];
            const auto rz = (x - volMeta->Origin[2]) / volMeta->SliceDists[2][0];

            if (rx > 0 && rx < volMeta->Resolution[0] && ry > 0 && ry < volMeta->Resolution[1] && rz > 0 &&
                rz < volMeta->Resolution[2]) {

                const int quantX = static_cast<int>(rx);
                const int quantY = static_cast<int>(ry);
                const int quantZ = static_cast<int>(rz);
                const float diffX = rx - quantX;
                const float diffY = ry - quantY;
                const float diffZ = rz - quantZ;

                const float c000 = inVol->GetRelativeVoxelValue(quantX, quantY, quantZ);
                const float c100 = inVol->GetRelativeVoxelValue(quantX + 1, quantY, quantZ);
                const float c010 = inVol->GetRelativeVoxelValue(quantX, quantY + 1, quantZ);
                const float c110 = inVol->GetRelativeVoxelValue(quantX + 1, quantY + 1, quantZ);
                const float c001 = inVol->GetRelativeVoxelValue(quantX, quantY, quantZ + 1);
                const float c101 = inVol->GetRelativeVoxelValue(quantX + 1, quantY, quantZ + 1);
                const float c011 = inVol->GetRelativeVoxelValue(quantX, quantY + 1, quantZ + 1);
                const float c111 = inVol->GetRelativeVoxelValue(quantX + 1, quantY + 1, quantZ + 1);

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
                    if (isInterleaved) {
                        memcpy(theVertexData[i].data() + vdstride * cntLeft,
                            commonBasePointer + vdstride * j, vdstride);
                    } else {
                        memcpy(theVertexData[i].data() + vdsize * cntLeft,
                            vertexBasePointer + vdstride * j, vdsize);
                        memcpy(theColorData[i].data() + cdsize * cntLeft,
                            colorBasePointer + cdstride * j, cdsize);
                    }
                    cntLeft++;
                }
            }
        }

        auto& outp = outData.AccessParticles(i);
        outp.SetCount(cntLeft);
        vislib::sys::Log::DefaultLog.WriteInfo(
            "ParticleVisibilityFromVolume: list %d: %lu / %lu particles left", i, cntLeft, cnt);
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

    this->lastTime = inData.FrameID();
    this->lastParticleHash = inData.DataHash();
    this->lastVolumeHash = inVol->DataHash();

    operatorSlot.ResetDirty();
    valueSlot.ResetDirty();
    epsilonSlot.ResetDirty();

    return true;
}
