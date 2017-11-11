/*
 * ParticleVelocities.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleVelocities.h"
#include "mmcore/param/BoolParam.h"
#include "nanoflann.hpp"
#include "vislib/sys/Log.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/ShallowPoint.h"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>

using namespace megamol;
using namespace megamol::stdplugin;

inline vislib::math::Vector<float, 3> getDifference(const float p1[3], const float p2[3],
    bool cyclicX, bool cyclicY, bool cyclicZ, 
    float width, float height, float depth) {
    float dx, dy, dz;
    
    // dr = np.remainder(r1 - r2 + L/2., L) - L/2.
    // remainder = x1 - floor(x1 / x2) * x2
    if (cyclicX) {
        float x1 = p1[0] - p2[0] + width / 2;
        dx = x1 - floor(x1 / width) * width;
        dx -= width / 2;
    } else {
        dx = p2[0] - p1[0];
    }
    if (cyclicY) {
        float y1 = p1[1] - p2[1] + height / 2;
        dy = y1 - floor(y1 / height) * height;
        dy -= height / 2;
    } else {
        dy = p2[1] - p1[1];
    }
    if (cyclicZ) {
        float z1 = p1[2] - p2[2] + depth / 2;
        dz = z1 - floor(z1 / depth) * depth;
        dz -= depth / 2;
    } else {
        dz = p2[2] - p1[2];
    }

    return vislib::math::Vector<float, 3>(dx, dy, dz);
}


/*
* datatools::ParticleVelocities::create
*/
bool datatools::ParticleVelocities::create(void) {
    return true;
}


/*
* datatools::ParticleVelocities::release
*/
void datatools::ParticleVelocities::release(void) {
}


/*
 * datatools::ParticleVelocities::ParticleVelocities
 */
datatools::ParticleVelocities::ParticleVelocities(void)
        : cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction"),
        cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction"),
        cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction"),
        outDataSlot("outData", "Provides one frame less than the source, but with velocities"),
        inDataSlot("inData", "Takes the particle data, sorted, with constant particle numbers over all frames"),
        cachedVertexData(), cachedNumLists(0), cachedTime(-1), cachedDirData(),
        datahash(0), time(0) {

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &ParticleVelocities::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleVelocities::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticleVelocities::~ParticleVelocities
 */
datatools::ParticleVelocities::~ParticleVelocities(void) {
    this->Release();
}


bool datatools::ParticleVelocities::assertData(core::moldyn::MultiParticleDataCall *in,
    core::moldyn::DirectionalParticleDataCall *out) {

    using megamol::core::moldyn::MultiParticleDataCall;

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
        if (!(*in)(1)) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleVelocities: could not get previous frame (%u)", time - 1);
            return false;
        }
        cachedVertexData.resize(in->GetParticleListCount(), nullptr);
        cachedVertexDataType.resize(in->GetParticleListCount());
        cachedGlobalRadius.resize(in->GetParticleListCount(), 0.0f);
        cachedDirData.resize(in->GetParticleListCount());
        cachedStride.resize(in->GetParticleListCount());
        for (unsigned int i = 0; i < in->GetParticleListCount(); i++) {
            size_t stride = in->AccessParticles(i).GetVertexDataStride();
            this->cachedVertexDataType[i] = in->AccessParticles(i).GetVertexDataType();
            if (stride == 0) {
                switch (this->cachedVertexDataType[i]) {
                    case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_NONE:
                        continue;
                    case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ:
                        cachedGlobalRadius[i] = in->AccessParticles(i).GetGlobalRadius();
                        stride = 12;
                        break;
                    case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR:
                        stride = 16;
                        break;
                    case MultiParticleDataCall::Particles::VertexDataType::VERTDATA_SHORT_XYZ:
                        vislib::sys::Log::DefaultLog.WriteError("ParticleVelocities: cannot process short position vertices");
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
        this->cachedTime = time - 1;
    }
    in->SetFrameID(time, true);
    if (!(*in)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleVelocities: could not get current frame (%u)", time);
        return false;
    }
    if (cachedNumLists != in->GetParticleListCount()) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleVelocities: inconsistent number of lists"
            "between frames %u (%u) and %u (%u)", time - 1, cachedNumLists, time, in->GetParticleListCount());
        return false;
    }

    bool cycleX = this->cyclXSlot.Param<core::param::BoolParam>()->Value();
    bool cycleY = this->cyclYSlot.Param<core::param::BoolParam>()->Value();
    bool cycleZ = this->cyclZSlot.Param<core::param::BoolParam>()->Value();

    out->SetParticleListCount(cachedNumLists);
    float *cachedPtr, *currentPtr;
    vislib::math::Vector<float, 3> diff;
    for (auto i = 0; i < cachedNumLists; i++) {
        if (cachedVertexDataType[i] != in->AccessParticles(i).GetVertexDataType()) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleVelocities: inconsistent vertex data type"
                "between frames %u (%u) and %u (%u)in list %u", time - 1, cachedVertexDataType[i],
                time, in->AccessParticles(i).GetVertexDataType(), i);
            return false;
        }
        if (cachedListLength[i] != in->AccessParticles(i).GetCount()) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleVelocities: inconsistent list length"
                "between frames %u (%u) and %u (%u)in list %u", time - 1, cachedListLength[i],
                time, in->AccessParticles(i).GetCount(), i);
            return false;
        }
        out->AccessParticles(i).SetCount(this->cachedListLength[i]);
        out->AccessParticles(i).SetVertexData(this->cachedVertexDataType[i], in->AccessParticles(i).GetVertexData(),
            in->AccessParticles(i).GetVertexDataStride());

        auto &bbox = in->GetBoundingBoxes().ObjectSpaceBBox();
        this->cachedDirData[i] = new float[cachedListLength[i] * 3];
        for (auto p = 0; p < this->cachedListLength[i]; p++) {
            cachedPtr = reinterpret_cast<float*>(static_cast<char*>(this->cachedVertexData[i]) + p * this->cachedStride[i]);
            currentPtr = reinterpret_cast<float*>(static_cast<char*>(const_cast<void*>(in->AccessParticles(i).GetVertexData()))
                + p * this->cachedStride[i]);
            diff = getDifference(cachedPtr, currentPtr, cycleX, cycleY, cycleZ, bbox.Width(), bbox.Height(), bbox.Depth());
            this->cachedDirData[i][p * 3 + 0] = diff[0];
            this->cachedDirData[i][p * 3 + 1] = diff[1];
            this->cachedDirData[i][p * 3 + 2] = diff[2];
        }

        out->AccessParticles(i).SetDirData(megamol::core::moldyn::DirectionalParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
            cachedDirData[i], 0);
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(in->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(in->GetBoundingBoxes().ObjectSpaceClipBox());
    return true;
}


bool datatools::ParticleVelocities::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&c);
    if (outDpdc == NULL) return false;

    MultiParticleDataCall *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == NULL) return false;

    if (!this->assertData(inMpdc, outDpdc)) return false;

    inMpdc->Unlock();

    return true;
}

bool datatools::ParticleVelocities::getDataCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&c);
    if (outDpdc == NULL) return false;

    MultiParticleDataCall *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == NULL) return false;

    if (!this->assertData(inMpdc, outDpdc)) return false;

    inMpdc->Unlock();

    return true;
}