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
#include <chrono>
#include <functional>
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::stdplugin;

/*
 * datatools::ParticlesToDensity::create
 */
bool datatools::ParticlesToDensity::create(void) { return true; }


/*
 * datatools::ParticlesToDensity::release
 */
void datatools::ParticlesToDensity::release(void) {
    delete[] this->metadata.MinValues;
    delete[] this->metadata.MaxValues;
    delete[] this->metadata.SliceDists[0];
    delete[] this->metadata.SliceDists[1];
    delete[] this->metadata.SliceDists[2];
}


/*
 * datatools::ParticlesToDensity::ParticlesToDensity
 */
datatools::ParticlesToDensity::ParticlesToDensity(void)
    : aggregatorSlot("aggregator", "algorithm for the aggregation")
    , xResSlot("sizex", "The size of the volume in numbers of voxels")
    , yResSlot("sizey", "The size of the volume in numbers of voxels")
    , zResSlot("sizez", "The size of the volume in numbers of voxels")
    , cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction")
    , cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction")
    , cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction")
    , normalizeSlot("normalize", "Normalize the output volume")
    , sigmaSlot("sigma", "Sigma for Gauss in multiple of rad")
    //, datahash(std::numeric_limits<size_t>::max())
    , datahash(0)
    , time(std::numeric_limits<unsigned int>::max())
    , outDataSlot("outData", "Provides a density volume for the particles")
    , inDataSlot("inData", "takes the particle data") {

    auto* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "PosToSingleCell_Volume");
    ep->SetTypePair(1, "IColToSingleCell_Volume");
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

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);
    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);
    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->normalizeSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->normalizeSlot);

    this->sigmaSlot << new core::param::FloatParam(1.0f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->sigmaSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticlesToDensity::~ParticlesToDensity
 */
datatools::ParticlesToDensity::~ParticlesToDensity(void) { this->Release(); }


bool datatools::ParticlesToDensity::getExtentCallback(megamol::core::Call& c) {
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

    inMpdc->SetFrameID(outVol->FrameID(), true);
    if (!(*inMpdc)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticlesToDensity: Unable to get extents.");
        return false;
    }
    if (!(*inMpdc)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticlesToDensity: Unable to get data.");
        return false;
    }
    if (this->time != inMpdc->FrameID() || this->in_datahash != inMpdc->DataHash() || this->anythingDirty()) {
        if (!this->createVolumeCPU(inMpdc)) return false;
        this->time = inMpdc->FrameID();
        this->in_datahash = inMpdc->DataHash();
        ++this->datahash;
        this->resetDirty();
    }

    // TODO set data
    outVol->SetData(this->vol[0].data());
    metadata.Components = 1;
    metadata.GridType = core::misc::GridType_t::CARTESIAN;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = core::misc::ScalarType_t::FLOATING_POINT;
    metadata.ScalarLength = sizeof(float);
    metadata.MinValues = new double[1];
    metadata.MinValues[0] = this->minDens;
    metadata.MaxValues = new double[1];
    metadata.MaxValues[0] = this->maxDens;
    auto bbox = inMpdc->AccessBoundingBoxes().ObjectSpaceBBox();
    metadata.Extents[0] = bbox.Width();
    metadata.Extents[1] = bbox.Height();
    metadata.Extents[2] = bbox.Depth();
    metadata.NumberOfFrames = 1;
    metadata.SliceDists[0] = new float[1];
    metadata.SliceDists[0][0] = metadata.Extents[0] / static_cast<float>(metadata.Resolution[0] - 1);
    metadata.SliceDists[1] = new float[1];
    metadata.SliceDists[1][0] = metadata.Extents[1] / static_cast<float>(metadata.Resolution[1] - 1);
    metadata.SliceDists[2] = new float[1];
    metadata.SliceDists[2][0] = metadata.Extents[2] / static_cast<float>(metadata.Resolution[2] - 1);

    metadata.Origin[0] = bbox.Left();
    //-metadata.SliceDists[0][0] / 4.0f;
    metadata.Origin[1] = bbox.Bottom();
    //-metadata.SliceDists[1][0] / 4.0f;
    metadata.Origin[2] = bbox.Back();
    //-metadata.SliceDists[2][0] / 4.0f;

    metadata.IsUniform[0] = true;
    metadata.IsUniform[1] = true;
    metadata.IsUniform[2] = true;
    outVol->SetMetadata(&metadata);

    outVol->SetDataHash(this->datahash);

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

    vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: starting volume creation");
    const auto startTime = std::chrono::high_resolution_clock::now();
    size_t totalParticles = 0;

    auto const sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    auto const sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    auto const sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    vol.resize(omp_get_max_threads());
    //std::vector<std::vector<unsigned int>> weights(omp_get_max_threads());
    int init, j;
#pragma omp parallel for
    for (init = 0; init < omp_get_max_threads(); init++) {
        vol[init].resize(sx * sy * sz, 0);
        //weights[init].resize(sx * sy * sz, 0);
    }

    // TODO: the whole code is wrong since we might not have the bounding box for the actual
    // cyclic boundary conditions. Also, these CBC are not applied currently.

    // TODO: what about near-zero or zero radii? This currently blows the whole thing up.

    bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();

    auto const minOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    auto const minOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    auto const minOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    auto const rangeOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Width();
    auto const rangeOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Height();
    auto const rangeOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Depth();
    auto const halfRangeOSx = c2->AccessBoundingBoxes().ObjectSpaceBBox().Width() * 0.5;
    auto const halfRangeOSy = c2->AccessBoundingBoxes().ObjectSpaceBBox().Height() * 0.5;
    auto const halfRangeOSz = c2->AccessBoundingBoxes().ObjectSpaceBBox().Depth() * 0.5;


    float const sliceDistX = rangeOSx / static_cast<float>(sx - 1);
    float const sliceDistY = rangeOSy / static_cast<float>(sy - 1);
    float const sliceDistZ = rangeOSz / static_cast<float>(sz - 1);

    float const maxCellSize = std::max(sliceDistX, std::max(sliceDistY, sliceDistZ));

    for (unsigned int i = 0; i < c2->GetParticleListCount(); ++i) {
        megamol::core::moldyn::MultiParticleDataCall::Particles& parts = c2->AccessParticles(i);
        const float globRad = parts.GetGlobalRadius();
        const bool useGlobRad =
            (parts.GetVertexDataType() ==
                megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) ||
            (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }

        totalParticles += parts.GetCount();

        // https : // en.wikipedia.org/wiki/Radial_basis_function
        auto gauss = [](float const dist, float const epsilon) -> float {
            return std::exp(-(dist * epsilon) * (dist * epsilon));
        };

        auto const& parStore = parts.GetParticleStore();
        auto const& xAcc = parStore.GetXAcc();
        auto const& yAcc = parStore.GetYAcc();
        auto const& zAcc = parStore.GetZAcc();
        auto const& rAcc = parStore.GetRAcc();
        auto const& iAcc = parStore.GetCRAcc();

        std::function<void(int, int, int, int, float, float)> volOp;
        switch (this->aggregatorSlot.Param<core::param::EnumParam>()->Value()) {
        case 1: {
            volOp = [this, &gauss, &iAcc, &sx, &sy](int const pidx, int const x, int const y, int const z,
                        float const dis, float const rad) -> void {
                auto const val = iAcc->Get_f(pidx);
                vol[omp_get_thread_num()][x + (y + z * sy) * sx] +=
                    gauss(dis, this->sigmaSlot.Param<core::param::FloatParam>()->Value() * rad) * val;
            };
        } break;
        default:
        case 0: {
            volOp = [this, &gauss, &sx, &sy](int const pidx, int const x, int const y, int const z,
                        float const dis, float const rad) -> void {
                vol[omp_get_thread_num()][x + (y + z * sy) * sx] +=
                    gauss(dis, this->sigmaSlot.Param<core::param::FloatParam>()->Value() * rad);
            };
        }
        }

#pragma omp parallel for
        for (int j = 0; j < parts.GetCount(); ++j) {
            auto const x_base = xAcc->Get_f(j);
            auto x = static_cast<int>((x_base - minOSx) / sliceDistX);
            if (x < 0)
                x = 0;
            else if (x >= sx)
                x = sx - 1;
            auto const y_base = yAcc->Get_f(j);
            auto y = static_cast<int>((y_base - minOSy) / sliceDistY);
            if (y < 0)
                y = 0;
            else if (y >= sy)
                y = sy - 1;
            auto const z_base = zAcc->Get_f(j);
            auto z = static_cast<int>((z_base - minOSz) / sliceDistZ);
            if (z < 0)
                z = 0;
            else if (z >= sz)
                z = sz - 1;
            auto rad = globRad;
            if (!useGlobRad) rad = rAcc->Get_f(j);

            int filterSizeX = static_cast<int>(std::ceilf(rad / sliceDistX));
            int filterSizeY = static_cast<int>(std::ceilf(rad / sliceDistY));
            int filterSizeZ = static_cast<int>(std::ceilf(rad / sliceDistZ));

            for (int hz = z - filterSizeZ; hz <= z + filterSizeZ; ++hz) {
                for (int hy = y - filterSizeY; hy <= y + filterSizeY; ++hy) {
                    for (int hx = x - filterSizeX; hx <= x + filterSizeX; ++hx) {
                        auto tmp_hx = hx;
                        auto tmp_hy = hy;
                        auto tmp_hz = hz;
                        if (cycl_x) {
                            tmp_hx = (hx + 2 * sx) % sx;
                        } else {
                            if (hx < 0 || hx > sx - 1) {
                                continue;
                            }
                        }
                        if (cycl_y) {
                            tmp_hy = (hy + 2 * sy) % sy;
                        } else {
                            if (hy < 0 || hy > sy - 1) {
                                continue;
                            }
                        }
                        if (cycl_z) {
                            tmp_hz = (hz + 2 * sz) % sz;
                        } else {
                            if (hz < 0 || hz > sz - 1) {
                                continue;
                            }
                        }

                        float x_diff = static_cast<float>(tmp_hx) * sliceDistX + minOSx;
                        x_diff = std::fabs(x_diff - x_base);
                        if (x_diff > halfRangeOSx) x_diff -= rangeOSx;
                        float y_diff = static_cast<float>(tmp_hy) * sliceDistY + minOSy;
                        y_diff = std::fabs(y_diff - y_base);
                        if (y_diff > halfRangeOSy) y_diff -= rangeOSy;
                        float z_diff = static_cast<float>(tmp_hz) * sliceDistZ + minOSz;
                        z_diff = std::fabs(z_diff - z_base);
                        if (z_diff > halfRangeOSz) z_diff -= rangeOSz;
                        float const dis = std::sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                        //if (dis == 0.0f) dis = 1.0f;
                        //vol[omp_get_thread_num()][hx + (hy + hz * sy) * sx] += 1.0f / dis;
                        
                        volOp(j, x, y, z, dis, rad);
                        
                        /*if (dis > disThreshold - rad) {
                            vol[omp_get_thread_num()][x + (y + z * sy) * sx] +=
                                gauss(dis - disThreshold + rad, 3.0f * rad);
                        } else {
                            vol[omp_get_thread_num()][x + (y + z * sy) * sx] += 1.0f;
                        }*/



                        //++weights[omp_get_thread_num()][hx + (hy + hz * sy) * sx];
                    }
                }
            }
        }
    }

    std::vector<float> localMax(omp_get_max_threads(), 0.0f);
    std::vector<float> localMin(omp_get_max_threads(), std::numeric_limits<float>::max());

    int i = 0;
#pragma omp parallel for private(i)
    for (j = 0; j < sx * sy * sz; ++j) {
        for (i = 1; i < omp_get_max_threads(); ++i) {
            vol[0][j] += vol[i][j];
            //weights[0][j] += weights[i][j];
            /*if (vol[i][j] > 0.0f) {
                vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: Thread %d found value != 0 in
            vol[%d][%d]\n", omp_get_thread_num(), i, j);
            }*/
        }
        //vol[0][j] /= weights[0][j] == 0 ? 1 : static_cast<float>(weights[0][j]);
        //vol[0][j] /= static_cast<float>(weights[0][j]);
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

//#define PTD_DEBUG_OUTPUT
#ifdef PTD_DEBUG_OUTPUT
    std::ofstream raw_file{"lasercross.raw", std::ios::binary};
    raw_file.write(reinterpret_cast<char const*>(vol[0].data()), vol[0].size() * sizeof(float));
    raw_file.close();
    vislib::sys::Log::DefaultLog.WriteInfo("ParticlesToDensity: Debug file written\n");
#endif

    // Cleanup
    vol.resize(1);

    const auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diffMillis = endTime - startTime;
    vislib::sys::Log::DefaultLog.WriteInfo(
        "ParticlesToDensity: creation of %u x %u x %u volume from %llu particles took %f ms.", sx, sy, sz,
        totalParticles, diffMillis.count());

    return true;
}


bool datatools::ParticlesToDensity::dummyCallback(megamol::core::Call& c) { return true; }
