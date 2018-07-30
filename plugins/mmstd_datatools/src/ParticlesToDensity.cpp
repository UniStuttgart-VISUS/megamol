/*
 * ParticlesToDensity.h
 *
 * Copyright (C) 2018 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticlesToDensity.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>
#include "mmcore/param/IntParam.h"
#include "omp.h"
#define _USE_MATH_DEFINES
#include <math.h>

using namespace megamol;
using namespace megamol::stdplugin;

/*
* datatools::ParticlesToDensity::create
*/
bool datatools::ParticlesToDensity::create(void) {
    return true;
}


/*
* datatools::ParticlesToDensity::release
*/
void datatools::ParticlesToDensity::release(void) {}


/*
 * datatools::ParticlesToDensity::ParticlesToDensity
 */
datatools::ParticlesToDensity::ParticlesToDensity(void)
    : aggregatorSlot("aggregator", "algorithm for the aggregation")
    , outDataSlot("outData", "Provides a density volume for the particles")
    , inDataSlot("inData", "takes the particle data") 
    , xResSlot("sizex", "The size of the volume in numbers of voxels")
    , yResSlot("sizey", "The size of the volume in numbers of voxels")
    , zResSlot("sizez", "The size of the volume in numbers of voxels")
    , datahash(0)
    , time(0) {

    auto* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "PosToSingleCell_Volume");
    this->aggregatorSlot << ep;
    this->MakeSlotAvailable(&this->aggregatorSlot);


    this->outDataSlot.SetCallback(core::moldyn::VolumeDataCall::ClassName(),
        core::moldyn::VolumeDataCall::FunctionName(core::moldyn::VolumeDataCall::CallForGetData),
        &ParticlesToDensity::getDataCallback);
    this->outDataSlot.SetCallback(core::moldyn::VolumeDataCall::ClassName(),
        core::moldyn::VolumeDataCall::FunctionName(core::moldyn::VolumeDataCall::CallForGetExtent),
        &ParticlesToDensity::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->xResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->xResSlot);

    this->yResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->yResSlot);

    this->zResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->zResSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticlesToDensity::~ParticlesToDensity
 */
datatools::ParticlesToDensity::~ParticlesToDensity(void) {
    this->Release();
}


bool datatools::ParticlesToDensity::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    auto* out = dynamic_cast<core::moldyn::VolumeDataCall*>(&c);
    if (out == nullptr) return false;

    auto *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == nullptr) return false;

    //if (!this->assertData(inMpdc, outDpdc)) return false;
    inMpdc->SetFrameID(out->FrameID(), true);
    if (!(*inMpdc)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticlesToDensity: could not get current frame extents (%u)", time - 1);
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(inMpdc->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(inMpdc->GetBoundingBoxes().ObjectSpaceClipBox());
    out->SetFrameCount(inMpdc->FrameCount());
    // TODO: what am I actually doing here
    //inMpdc->SetUnlocker(nullptr, false);
    //inMpdc->Unlock();

    return true;
}

bool datatools::ParticlesToDensity::getDataCallback(megamol::core::Call& c) {

    auto *inMpdc = this->inDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inMpdc == nullptr) return false;

    auto* outVol = dynamic_cast<core::moldyn::VolumeDataCall*>(&c);
    if (outVol == nullptr) return false;

    if (!(*inMpdc)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticlesToDensity: Unable to get data.");
        return false;
    }
    if (this->time != outVol->FrameID() || this->datahash != inMpdc->DataHash()) {
        if (!this->createVolumeCPU(inMpdc)) return false;
        this->time = outVol->FrameID();
        this->datahash = inMpdc->DataHash();
    }

    // TODO set data
    outVol->SetVolumeDimension(this->xResSlot.Param<core::param::IntParam>()->Value(),
        this->yResSlot.Param<core::param::IntParam>()->Value(), this->zResSlot.Param<core::param::IntParam>()->Value());
    outVol->SetComponents(1);
    outVol->SetMinimumDensity(0.0f);
    outVol->SetMaximumDensity(this->maxDens);
    outVol->SetVoxelMapPointer(this->vol[0].data());
    //inMpdc->Unlock();

    return true;
}


/*
 *  moldyn::DynDensityGradientEstimator::createVolumeCPU
 */
bool datatools::ParticlesToDensity::createVolumeCPU(class megamol::core::moldyn::MultiParticleDataCall* c2) {

    int sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    int sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    int sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    vol.resize(omp_get_max_threads());
    int init, j;
#pragma omp parallel for
    for (init = 0; init < omp_get_max_threads(); init++) {
        vol[init].resize(sx * sy * sz, 0);
    }

    float minOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    float minOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    float minOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    float rangeOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Width();
    float rangeOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Height();
    float rangeOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Depth();

    // float volGenFac = this->aoGenFacSlot.Param<megamol::core::param::FloatParam>()->Value();
    float volGenFac = 1.0f;
    //    float voxelVol = (rangeOSx / static_cast<float>(sx))
    //        * (rangeOSy / static_cast<float>(sy))
    //        * (rangeOSz / static_cast<float>(sz));
    const float voxelVol = (rangeOSx / static_cast<float>(sx - 1)) * (rangeOSy / static_cast<float>(sy - 1)) *
                     (rangeOSz / static_cast<float>(sz - 1));

    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles& parts = c2->AccessParticles(i);
        const float globRad = parts.GetGlobalRadius();
        const bool useGlobRad =
            (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) ||
            (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }

        const float globSpVol = 4.0f / 3.0f * static_cast<float>(M_PI) * globRad * globRad * globRad;

#pragma omp parallel for
        for (j = 0; j < parts.GetCount(); j++) {
            //const float* ppos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride * j);
            auto ppos = parts[j];
            int x = static_cast<int>(((ppos.vert.GetXf() - minOSx) / rangeOSx) * static_cast<float>(sx - 1));
            if (x < 0)
                x = 0;
            else if (x >= sx)
                x = sx - 1;
            int y = static_cast<int>(((ppos.vert.GetYf() - minOSy) / rangeOSy) * static_cast<float>(sy - 1));
            if (y < 0)
                y = 0;
            else if (y >= sy)
                y = sy - 1;
            int z = static_cast<int>(((ppos.vert.GetZf() - minOSz) / rangeOSz) * static_cast<float>(sz - 1));
            if (z < 0)
                z = 0;
            else if (z >= sz)
                z = sz - 1;
            float spVol = globSpVol;
            if (!useGlobRad) {
                const float rad = ppos.vert.GetRf();
                spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;
            }
            vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (spVol / voxelVol) * volGenFac;
        }
    }

    std::vector<float> localMax(omp_get_max_threads());

#pragma omp parallel for
    for (j = 0; j < sx * sy * sz; j++) {
        for (int i = 1; i < omp_get_max_threads(); i++) {
            vol[0][j] += vol[i][j];
        }
        if (vol[0][j] > localMax[omp_get_thread_num()]) localMax[omp_get_thread_num()] = vol[0][j];
    }

    maxDens = *std::max_element(localMax.begin(), localMax.end());

    // Cleanup
    vol.resize(1);

    return true;
}
