/*
 * ParticlesToDensity.h
 *
 * Copyright (C) 2018 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include "ParticlesToDensity.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
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
    , xResSlot("sizex", "The size of the volume in numbers of voxels")
    , yResSlot("sizey", "The size of the volume in numbers of voxels")
    , zResSlot("sizez", "The size of the volume in numbers of voxels")
    , normalizeSlot("normalize", "Normalize the output volume")
    , datahash(0)
    , time(0)
    , outDataSlot("outData", "Provides a density volume for the particles")
    , inDataSlot("inData", "takes the particle data") {

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
        &ParticlesToDensity::getExtentCallback);
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

    this->normalizeSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->normalizeSlot);

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
    metadata.MinValues[0] = this->minDens;
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
    metadata.SliceDists[0][0] = metadata.Extents[0] / static_cast<float>(metadata.Resolution[0] - 1);
    metadata.SliceDists[1] = new float;
    metadata.SliceDists[1][0] = metadata.Extents[1] / static_cast<float>(metadata.Resolution[1] - 1);
    metadata.SliceDists[2] = new float;
    metadata.SliceDists[2][0] = metadata.Extents[2] / static_cast<float>(metadata.Resolution[2] - 1);
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

    auto const sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    auto const sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    auto const sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    vol.resize(omp_get_max_threads());
    int init, j;
#pragma omp parallel for
    for (init = 0; init < omp_get_max_threads(); init++) {
        vol[init].resize(sx * sy * sz, 0);
    }

    auto const minOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    auto const minOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    auto const minOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    auto const rangeOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Width();
    auto const rangeOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Height();
    auto const rangeOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Depth();

    //    float voxelVol = (rangeOSx / static_cast<float>(sx))
    //        * (rangeOSy / static_cast<float>(sy))
    //        * (rangeOSz / static_cast<float>(sz));
    const float voxelVol = (rangeOSx / static_cast<float>(sx - 1)) * (rangeOSy / static_cast<float>(sy - 1)) *
                           (rangeOSz / static_cast<float>(sz - 1));

    float const cellSizex = rangeOSx / static_cast<float>(sx - 1);
    float const cellSizey = rangeOSy / static_cast<float>(sy - 1);
    float const cellSizez = rangeOSz / static_cast<float>(sz - 1);

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

        auto gauss = [](float x, float y, float z, float sigma) -> float {
            return std::expf(-(x * x + y * y + z * z) / (2.0f * sigma * sigma)) /
                   std::powf(std::sqrtf(2.0f * M_PI * sigma), 3);
        };

        int filterSize = 1;

#pragma omp parallel for
        for (int j = 0; j < parts.GetCount(); ++j) {
            auto ppos = parts[j];
            auto const x_base = ppos.vert.GetXf();
            int x = static_cast<int>(((x_base - minOSx) / rangeOSx) * static_cast<float>(sx - 1));
            if (x < 0)
                x = 0;
            else if (x >= sx)
                x = sx - 1;
            auto const y_base = ppos.vert.GetYf();
            int y = static_cast<int>(((y_base - minOSy) / rangeOSy) * static_cast<float>(sy - 1));
            if (y < 0)
                y = 0;
            else if (y >= sy)
                y = sy - 1;
            auto const z_base = ppos.vert.GetZf();
            int z = static_cast<int>(((z_base - minOSz) / rangeOSz) * static_cast<float>(sz - 1));
            if (z < 0)
                z = 0;
            else if (z >= sz)
                z = sz - 1;

            for (int hz = z - filterSize; hz <= z + filterSize; ++hz) {
                for (int hy = y - filterSize; hy <= y + filterSize; ++hy) {
                    for (int hx = x - filterSize; hx <= x + filterSize; ++hx) {
                        if (hx >= 0 && hx < sx && hy >= 0 && hy < sy && hz >= 0 && hz < sz) {
                            float x_diff =
                                static_cast<float>(hx) / static_cast<float>(sx - 1) * rangeOSx + minOSx + 0.5f * cellSizex;
                            x_diff = std::fabsf(x_diff - x_base);
                            float y_diff =
                                static_cast<float>(hy) / static_cast<float>(sy - 1) * rangeOSy + minOSy + 0.5f * cellSizey;
                            y_diff = std::fabsf(y_diff - y_base);
                            float z_diff =
                                static_cast<float>(hz) / static_cast<float>(sz - 1) * rangeOSz + minOSz + 0.5f * cellSizez;
                            z_diff = std::fabsf(z_diff - z_base);
                            float dis = std::sqrtf(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                            if (dis == 0.0f) dis = 1.0f;
                            vol[omp_get_thread_num()][hx + (hy + hz * sy) * sx] += 1.0f / dis;
                        }
                    }
                }
            }
        }

#if 0
#pragma omp parallel for
        for (j = 0; j < parts.GetCount(); ++j) {
            // const float* ppos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride * j);
            auto ppos = parts[j];
            auto x_base = ppos.vert.GetXf();
            int x = static_cast<int>(((x_base - minOSx) / rangeOSx) * static_cast<float>(sx - 1));
            if (x < 0)
                x = 0;
            else if (x >= sx)
                x = sx - 1;
            auto y_base = ppos.vert.GetYf();
            int y = static_cast<int>(((y_base - minOSy) / rangeOSy) * static_cast<float>(sy - 1));
            if (y < 0)
                y = 0;
            else if (y >= sy)
                y = sy - 1;
            auto z_base = ppos.vert.GetZf();
            int z = static_cast<int>(((z_base - minOSz) / rangeOSz) * static_cast<float>(sz - 1));
            if (z < 0)
                z = 0;
            else if (z >= sz)
                z = sz - 1;

            //for (int hz = z - filterSize; hz <= z + filterSize; ++hz) {
            //    for (int hy = y - filterSize; hy <= y + filterSize; ++hy) {
            //        for (int hx = x - filterSize; hx <= x + filterSize; ++hx) {
            //            if (hx >= 0 && hx < sx && hy >= 0 && hy < sy && hz >= 0 && hz < sz) {
            //                float x_pos = x_base;
            //                if (hx < x)
            //                    x_pos =
            //                        static_cast<float>(hx) / static_cast<float>(sx - 1) * rangeOSx + minOSx + cellSizex;
            //                else if (hx > x)
            //                    x_pos = static_cast<float>(hx) / static_cast<float>(sx - 1) * rangeOSx + minOSx;
            //                float y_pos = y_base;
            //                if (hy < y)
            //                    y_pos =
            //                        static_cast<float>(hy) / static_cast<float>(sy - 1) * rangeOSy + minOSy + cellSizey;
            //                else if (hy > y)
            //                    y_pos = static_cast<float>(hy) / static_cast<float>(sy - 1) * rangeOSy + minOSy;
            //                float z_pos = z_base;
            //                if (hz < z)
            //                    z_pos =
            //                        static_cast<float>(hz) / static_cast<float>(sz - 1) * rangeOSz + minOSz + cellSizez;
            //                else if (hz > z)
            //                    z_pos = static_cast<float>(hz) / static_cast<float>(sz - 1) * rangeOSz + minOSz;
            //                float spRad = globRad;
            //                if (!useGlobRad) {
            //                    spRad = ppos.vert.GetRf();
            //                }
            //                auto g = gauss(std::fabsf(x_pos - x_base), std::fabsf(y_pos - y_base),
            //                    std::fabsf(z_pos - z_base), 5.0f*spRad);
            //                vol[omp_get_thread_num()][hx + (hy + hz * sy) * sx] = g;
            //                /*vol[0][hx + (hy + hz * sy) * sx] = g;
            //                vislib::sys::Log::DefaultLog.WriteInfo(
            //                    "ParticlesToDensity: pos %f %f %f gauss %f\n", x_pos, y_pos, z_pos, g);*/
            //            }
            //        }
            //    }
            //}

            ++vol[omp_get_thread_num()][x + (y + z * sy) * sx];

            /*float spVol = globSpVol;
            if (!useGlobRad) {
                const float rad = ppos.vert.GetRf();
                spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;
            }
            vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (spVol / voxelVol);*/
        }
#endif
    }

    std::vector<float> localMax(omp_get_max_threads(), 0.0f);
    std::vector<float> localMin(omp_get_max_threads(), std::numeric_limits<float>::max());

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
        if (vol[0][j] < localMin[omp_get_thread_num()]) {
            localMin[omp_get_thread_num()] = vol[0][j];
            // vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: Thread %d found a new max: %f\n",
            // omp_get_thread_num(), vol[0][j]);
        }
    }

//#define DEBUG_OUTPUT
#ifdef DEBUG_OUTPUT
    std::ofstream raw_file{"lasercross.raw", std::ios::binary};
    raw_file.write(reinterpret_cast<char const*>(vol[0].data()), vol[0].size() * sizeof(float));
    raw_file.close();
    vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: Debug file written\n");
#endif

    maxDens = *std::max_element(localMax.begin(), localMax.end());
    minDens = *std::min_element(localMin.begin(), localMin.end());

    if (this->normalizeSlot.Param<core::param::BoolParam>()->Value()) {
        auto const rcpValRange = 1.0f / (maxDens - minDens);
#pragma omp parallel for
        for (int64_t i = 0; i < vol[0].size(); ++i) {
            vol[0][i] -= minDens;
            vol[0][i] *= rcpValRange;
        }
        maxDens = 1.0f;
        minDens = 0.0f;
    }

    // Cleanup
    vol.resize(1);

    return true;
}


bool datatools::ParticlesToDensity::dummyCallback(megamol::core::Call& c) { return true; }
