/*
 * ParticleVelocities.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "ParticleVelocities.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <nanoflann.hpp>

using namespace megamol;

inline vislib::math::Vector<float, 3> getDifference(const float p1[3], const float p2[3], bool cyclicX, bool cyclicY,
    bool cyclicZ, float width, float height, float depth) {
    float dx, dy, dz;

    // dr = np.remainder(r1 - r2 + L/2., L) - L/2.
    // remainder = x1 - floor(x1 / x2) * x2
    dx = p2[0] - p1[0];
    if (cyclicX && fabs(dx) > width / 2) {
        float x1 = p1[0] - p2[0] + width / 2;
        dx = x1 - floor(x1 / width) * width;
        dx -= width / 2;
    }
    dy = p2[1] - p1[1];
    if (cyclicY && fabs(dy) > height / 2) {
        float y1 = p1[1] - p2[1] + height / 2;
        dy = y1 - floor(y1 / height) * height;
        dy -= height / 2;
    }
    dz = p2[2] - p1[2];
    if (cyclicZ && fabs(dz) > depth / 2) {
        float z1 = p1[2] - p2[2] + depth / 2;
        dz = z1 - floor(z1 / depth) * depth;
        dz -= depth / 2;
    }

    return vislib::math::Vector<float, 3>(dx, dy, dz);
}


/*
 * datatools::ParticleVelocities::create
 */
bool datatools::ParticleVelocities::create() {
    return true;
}


/*
 * datatools::ParticleVelocities::release
 */
void datatools::ParticleVelocities::release() {}


/*
 * datatools::ParticleVelocities::ParticleVelocities
 */
datatools::ParticleVelocities::ParticleVelocities()
        : cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction")
        , cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction")
        , cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction")
        , dtSlot("dt", "time difference between two sequential time steps")
        , outDataSlot("outData", "Provides one frame less than the source, but with velocities")
        , inDataSlot("inData", "Takes the particle data, sorted, with constant particle numbers over all frames")
        , cachedVertexData()
        , cachedNumLists(0)
        , cachedTime(-1)
        , cachedDirData()
        , datahash(0)
        , time(0) {

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->dtSlot.SetParameter(new core::param::FloatParam(0.1f, 0.0000001f, 100.0f));
    this->MakeSlotAvailable(&this->dtSlot);

    this->outDataSlot.SetCallback(
        geocalls::MultiParticleDataCall::ClassName(), "GetData", &ParticleVelocities::getDataCallback);
    this->outDataSlot.SetCallback(
        geocalls::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleVelocities::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticleVelocities::~ParticleVelocities
 */
datatools::ParticleVelocities::~ParticleVelocities() {
    this->Release();
}


bool datatools::ParticleVelocities::assertData(
    geocalls::MultiParticleDataCall* in, geocalls::MultiParticleDataCall* outMPDC) {

    using geocalls::MultiParticleDataCall;

    megamol::core::AbstractGetData3DCall* out;
    if (outMPDC != nullptr)
        out = outMPDC;
    unsigned int time = out->FrameID() + 1; // we do not give out the original frame 0 because it has no previous frame

    if (this->cachedTime != time - 1 || this->datahash != in->DataHash()) {
        for (auto i = 0; i < cachedNumLists; i++) {
            if (cachedVertexDataType[i] != MultiParticleDataCall::Particles::VertexDataType::VERTDATA_NONE)
                delete this->cachedVertexData[i];
            delete this->cachedDirData[i];
            this->cachedListLength[i] = 0;
        }
        //cachedVertexData.resize(0);
        this->cachedTime = -1;
        this->cachedNumLists = 0;
        // load previous Frame
        in->SetFrameID(time - 1, true);
        //if (!(*in)(1)) {
        //    megamol::core::utility::log::Log::DefaultLog.WriteError("ParticleVelocities: could not get previous frame extents (%u)", time - 1);
        //    return false;
        //}
        do {
            if (!(*in)(0)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ParticleVelocities: could not get previous frame (%u)", time - 1);
                return false;
            }
        } while (in->FrameID() != time - 1); // did we get correct frame?
        cachedVertexData.resize(in->GetParticleListCount(), nullptr);
        cachedVertexDataType.resize(in->GetParticleListCount());
        cachedGlobalRadius.resize(in->GetParticleListCount(), 0.0f);
        cachedDirData.resize(in->GetParticleListCount());
        cachedStride.resize(in->GetParticleListCount());
        cachedListLength.resize(in->GetParticleListCount());
        for (unsigned int i = 0; i < in->GetParticleListCount(); i++) {
            size_t stride = in->AccessParticles(i).GetVertexDataStride();
            this->cachedVertexDataType[i] = in->AccessParticles(i).GetVertexDataType();
            cachedGlobalRadius[i] = in->AccessParticles(i).GetGlobalRadius();
            if (stride == 0) {
                switch (this->cachedVertexDataType[i]) {
                case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ:
                    stride = 12;
                    break;
                case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR:
                    stride = 16;
                    break;
                case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_SHORT_XYZ:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "ParticleVelocities: cannot process short position vertices");
                    this->cachedVertexDataType[i] = MultiParticleDataCall::Particles::VertexDataType::VERTDATA_NONE;
                    continue;
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "ParticleVelocities: cannot process unknown position type");
                    this->cachedVertexDataType[i] = MultiParticleDataCall::Particles::VertexDataType::VERTDATA_NONE;
                    continue;
                }
            }
            this->cachedStride[i] = stride;
            this->cachedListLength[i] = in->AccessParticles(i).GetCount();
            size_t thesize = this->cachedListLength[i] * stride;
            this->cachedVertexData[i] = new char[thesize];
            memcpy(this->cachedVertexData[i], in->AccessParticles(i).GetVertexData(), thesize);
        }
        // TODO: what am I actually doing here
        //in->SetUnlocker(nullptr, false);
        in->Unlock();
        this->cachedTime = time - 1;
        this->cachedNumLists = in->GetParticleListCount();

        in->SetFrameID(time, true);
        do {
            if (!(*in)(1)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ParticleVelocities: could not get current frame extents (%u)", time - 1);
                return false;
            }
            if (!(*in)(0)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ParticleVelocities: could not get current frame (%u)", time - 1);
                return false;
            }
        } while (in->FrameID() != time); // did we get correct frame?
        if (cachedNumLists != in->GetParticleListCount()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("ParticleVelocities: inconsistent number of lists"
                                                                    "between frames %u (%u) and %u (%u)",
                time - 1, cachedNumLists, time, in->GetParticleListCount());
            return false;
        }
        this->dtSlot.ForceSetDirty();
    }
    if (this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() || this->cyclZSlot.IsDirty() || this->dtSlot.IsDirty()) {
        bool cycleX = this->cyclXSlot.Param<core::param::BoolParam>()->Value();
        bool cycleY = this->cyclYSlot.Param<core::param::BoolParam>()->Value();
        bool cycleZ = this->cyclZSlot.Param<core::param::BoolParam>()->Value();
        float theDt = this->dtSlot.Param<core::param::FloatParam>()->Value();

        float *cachedPtr, *currentPtr;
        vislib::math::Vector<float, 3> diff;
        for (auto i = 0; i < cachedNumLists; i++) {
            if (cachedVertexDataType[i] != in->AccessParticles(i).GetVertexDataType()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ParticleVelocities: inconsistent vertex data type"
                    "between frames %u (%u) and %u (%u)in list %u",
                    time - 1, cachedVertexDataType[i], time, in->AccessParticles(i).GetVertexDataType(), i);
                return false;
            }
            if (cachedListLength[i] != in->AccessParticles(i).GetCount()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("ParticleVelocities: inconsistent list length"
                                                                        "between frames %u (%u) and %u (%u)in list %u",
                    time - 1, cachedListLength[i], time, in->AccessParticles(i).GetCount(), i);
                return false;
            }

            auto& bbox = in->GetBoundingBoxes().ObjectSpaceBBox();
            this->cachedDirData[i] = new float[cachedListLength[i] * 3];
            for (auto p = 0; p < this->cachedListLength[i]; p++) {
                cachedPtr =
                    reinterpret_cast<float*>(static_cast<char*>(this->cachedVertexData[i]) + p * this->cachedStride[i]);
                currentPtr = reinterpret_cast<float*>(
                    static_cast<char*>(const_cast<void*>(in->AccessParticles(i).GetVertexData())) +
                    p * this->cachedStride[i]);
                diff = getDifference(
                    cachedPtr, currentPtr, cycleX, cycleY, cycleZ, bbox.Width(), bbox.Height(), bbox.Depth());
                this->cachedDirData[i][p * 3 + 0] = diff[0] / theDt;
                this->cachedDirData[i][p * 3 + 1] = diff[1] / theDt;
                this->cachedDirData[i][p * 3 + 2] = diff[2] / theDt;
            }
        }
        // TODO: what am I actually doing here
        //in->SetUnlocker(nullptr, false);
        //in->Unlock();
    }
    if (outMPDC != nullptr) {
        outMPDC->SetParticleListCount(cachedNumLists);
        for (auto i = 0; i < cachedNumLists; i++) {
            outMPDC->AccessParticles(i).SetCount(this->cachedListLength[i]);
            outMPDC->AccessParticles(i).SetGlobalRadius(this->cachedGlobalRadius[i]);
            outMPDC->AccessParticles(i).SetVertexData(this->cachedVertexDataType[i],
                in->AccessParticles(i).GetVertexData(), in->AccessParticles(i).GetVertexDataStride());
            outMPDC->AccessParticles(i).SetColourData(in->AccessParticles(i).GetColourDataType(),
                in->AccessParticles(i).GetColourData(), in->AccessParticles(i).GetColourDataStride());
            outMPDC->AccessParticles(i).SetDirData(
                geocalls::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ, cachedDirData[i], 0);
        }
    }
    this->datahash = in->DataHash();
    out->SetUnlocker(in->GetUnlocker());
    in->SetUnlocker(nullptr, false);
    return true;
}


bool datatools::ParticleVelocities::getExtentCallback(megamol::core::Call& c) {
    using geocalls::MultiParticleDataCall;

    MultiParticleDataCall* outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == nullptr)
        return false;

    MultiParticleDataCall* inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == nullptr)
        return false;

    megamol::core::AbstractGetData3DCall* out;
    if (outMpdc != nullptr)
        out = outMpdc;

    //if (!this->assertData(inMpdc, outDpdc)) return false;
    inMpdc->SetFrameID(out->FrameID(), true);
    if (!(*inMpdc)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "ParticleVelocities: could not get current frame extents (%u)", time - 1);
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(inMpdc->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(inMpdc->GetBoundingBoxes().ObjectSpaceClipBox());
    if (inMpdc->FrameCount() < 2) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "ParticleVelocities: you cannot use this module for single-timestep data!");
        return false;
    }
    out->SetFrameCount(inMpdc->FrameCount() - 1);
    // TODO: what am I actually doing here
    inMpdc->SetUnlocker(nullptr, false);
    inMpdc->Unlock();

    return true;
}

bool datatools::ParticleVelocities::getDataCallback(megamol::core::Call& c) {
    using geocalls::MultiParticleDataCall;

    MultiParticleDataCall* outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == nullptr)
        return false;

    MultiParticleDataCall* inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == nullptr)
        return false;

    if (!this->assertData(inMpdc, outMpdc))
        return false;

    //inMpdc->Unlock();

    return true;
}
