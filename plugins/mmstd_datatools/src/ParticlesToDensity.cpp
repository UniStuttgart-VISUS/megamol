/*
 * ParticlesToDensity.h
 *
 * Copyright (C) 2018 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include "ParticlesToDensity.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "omp.h"
#include "vislib/sys/Log.h"
#define _USE_MATH_DEFINES
#include <math.h>

using namespace megamol;
using namespace megamol::stdplugin;

/*
 * datatools::ParticlesToDensity::create
 */
bool datatools::ParticlesToDensity::create(void) { return true; }


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


    /*this->outDataSlot.SetCallback(core::moldyn::VolumeDataCall::ClassName(),
        core::moldyn::VolumeDataCall::FunctionName(core::moldyn::VolumeDataCall::CallForGetData),
        &ParticlesToDensity::getDataCallback);
    this->outDataSlot.SetCallback(core::moldyn::VolumeDataCall::ClassName(),
        core::moldyn::VolumeDataCall::FunctionName(core::moldyn::VolumeDataCall::CallForGetExtent),
        &ParticlesToDensity::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);*/

    this->outDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_DATA),
        &ParticlesToDensity::getDataCallback);
    this->outDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_EXTENTS),
        &ParticlesToDensity::getExtentCallback);
    this->outDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_METADATA),
        &ParticlesToDensity::dummyCallback);
    this->outDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_START_ASYNC),
        &ParticlesToDensity::dummyCallback);
    this->outDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_STOP_ASYNC),
        &ParticlesToDensity::dummyCallback);
    this->outDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_TRY_GET_DATA),
        &ParticlesToDensity::dummyCallback);
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
datatools::ParticlesToDensity::~ParticlesToDensity(void) { this->Release(); }


bool datatools::ParticlesToDensity::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::DirectionalParticleDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    auto* out = dynamic_cast<core::misc::VolumetricDataCall*>(&c);
    if (out == nullptr) return false;

    auto* inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == nullptr) return false;

    // if (!this->assertData(inMpdc, outDpdc)) return false;
    inMpdc->SetFrameID(out->FrameID(), true);
    if (!(*inMpdc)(1)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "ParticlesToDensity: could not get current frame extents (%u)", time - 1);
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(inMpdc->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(inMpdc->GetBoundingBoxes().ObjectSpaceClipBox());
    out->SetFrameCount(inMpdc->FrameCount());
    // TODO: what am I actually doing here
    // inMpdc->SetUnlocker(nullptr, false);
    // inMpdc->Unlock();

    return true;
}

bool datatools::ParticlesToDensity::getDataCallback(megamol::core::Call& c) {

    auto* inMpdc = this->inDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inMpdc == nullptr) return false;

    auto* outVol = dynamic_cast<core::misc::VolumetricDataCall*>(&c);
    if (outVol == nullptr) return false;

    if (!(*inMpdc)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticlesToDensity: Unable to get extents.");
        return false;
    }
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
    outVol->SetData(this->vol[0].data());
    metadata.Components = 1;
    metadata.GridType = core::misc::GridType_t::RECTILINEAR;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = core::misc::ScalarType_t::FLOATING_POINT;
    metadata.ScalarLength = sizeof(float);
    metadata.MinValues = new double;
    metadata.MinValues[0] = 0.0f;
    metadata.MaxValues = new double;
    metadata.MaxValues[0] = this->maxDens;
    auto bbox = inMpdc->AccessBoundingBoxes().ObjectSpaceBBox();
    metadata.Extents[0] = bbox.Width();
    metadata.Extents[1] = bbox.Height();
    metadata.Extents[2] = bbox.Depth();
    metadata.Origin[0] = bbox.Left();
    metadata.Origin[1] = bbox.Bottom();
    metadata.Origin[2] = bbox.Back();
    metadata.NumberOfFrames = 1;
    metadata.SliceDists[0] = new float;
    metadata.SliceDists[0][0] = metadata.Extents[0] / static_cast<float>(metadata.Resolution[0]);
    metadata.SliceDists[1] = new float;
    metadata.SliceDists[1][0] = metadata.Extents[1] / static_cast<float>(metadata.Resolution[1]);
    metadata.SliceDists[2] = new float;
    metadata.SliceDists[2][0] = metadata.Extents[2] / static_cast<float>(metadata.Resolution[2]);
    metadata.IsUniform[0] = true;
    metadata.IsUniform[1] = true;
    metadata.IsUniform[2] = true;
    outVol->SetMetadata(&metadata);

    /*outVol->SetVolumeDimension(this->xResSlot.Param<core::param::IntParam>()->Value(),
        this->yResSlot.Param<core::param::IntParam>()->Value(), this->zResSlot.Param<core::param::IntParam>()->Value());
    outVol->SetComponents(1);
    outVol->SetMinimumDensity(0.0f);
    outVol->SetMaximumDensity(this->maxDens);
    outVol->SetVoxelMapPointer(this->vol[0].data());*/
    // inMpdc->Unlock();

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
            (parts.GetVertexDataType() ==
                megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) ||
            (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }

        const float globSpVol = 4.0f / 3.0f * static_cast<float>(M_PI) * globRad * globRad * globRad;

#pragma omp parallel for
        for (j = 0; j < parts.GetCount(); j++) {
            // const float* ppos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride * j);
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

    std::vector<float> localMax(omp_get_max_threads(), 0.0f);

    int i = 0;
#pragma omp parallel for private(i)
    for (j = 0; j < sx * sy * sz; ++j) {
        for (i = 1; i < omp_get_max_threads(); ++i) {
            vol[0][j] += vol[i][j];
            /*if (vol[i][j] > 0.0f) {
                vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: Thread %d found value != 0 in
            vol[%d][%d]\n", omp_get_thread_num(), i, j);
            }*/
        }
        if (vol[0][j] > localMax[omp_get_thread_num()]) {
            localMax[omp_get_thread_num()] = vol[0][j];
            // vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: Thread %d found a new max: %f\n",
            // omp_get_thread_num(), vol[0][j]);
        }
    }

    maxDens = *std::max_element(localMax.begin(), localMax.end());

    // Cleanup
    vol.resize(1);

    return true;
}


bool datatools::ParticlesToDensity::dummyCallback(megamol::core::Call &c) { return true; }

