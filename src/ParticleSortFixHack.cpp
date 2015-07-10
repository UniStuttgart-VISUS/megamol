/*
 * ParticleSortFixHack.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleSortFixHack.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include <cstdint>
#include <algorithm>
#include "vislib/sys/Log.h"
#include "vislib/sys/ConsoleProgressBar.h"
#include "vislib/sys/Thread.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleSortFixHack::ParticleSortFixHack
 */
datatools::ParticleSortFixHack::ParticleSortFixHack(void)
        : AbstractParticleManipulator("outData", "indata"), 
        data(), inDataTime(-1), outDataTime(0), 
        ids(), inDataHash(-1), outDataHash(0) {
}


/*
 * datatools::ParticleSortFixHack::~ParticleSortFixHack
 */
datatools::ParticleSortFixHack::~ParticleSortFixHack(void) {
    this->Release();
}


/*
 * datatools::ParticleSortFixHack::manipulateData
 */
bool datatools::ParticleSortFixHack::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    // update id table if requried
    if ((inData.DataHash() == 0) || (inData.DataHash() != inDataHash)) {
        inDataHash = inData.DataHash();
        inDataTime = -1; // force update of frame data

        unsigned int fid = inData.FrameID();
        bool fid_f = inData.IsFrameForced();

        if (!updateIDdata(inData)) return false;

        inData.SetFrameID(fid, fid_f);
        if (!inData(0)) return false;
    }

    // update frame data if required
    if ((inData.FrameID() != inDataTime)
            || (data.size() != inData.GetParticleListCount())) {
        inDataTime = inData.FrameID();
        updateData(inData);

    }

    // output internal data copy
    outData.SetDataHash(outDataHash);
    outData.SetExtent(inData.FrameCount(), inData.AccessBoundingBoxes());
    outData.SetFrameID(outDataTime);
    outData.SetParticleListCount(static_cast<unsigned int>(data.size()));
    for (unsigned int i = 0; i < static_cast<unsigned int>(data.size()); i++) {
        outData.AccessParticles(i) = data[i].parts;
    }
    outData.SetUnlocker(nullptr); // since this is a hack module, I don't care for thread safety

    return true;
}

bool datatools::ParticleSortFixHack::updateIDdata(megamol::core::moldyn::MultiParticleDataCall& inData) {
    outDataHash++;
    vislib::sys::Log::DefaultLog.WriteInfo("Updating Particle Sorting ID Data...");

    inData.SetFrameID(0, false);
    if (!inData(1)) return false; // query extends first for number for frames

    // bbox to resolve cyclic boundary conditions
    vislib::math::Cuboid<float> bbox = 
        inData.AccessBoundingBoxes().IsObjectSpaceBBoxValid()
        ? inData.AccessBoundingBoxes().ObjectSpaceBBox()
        : inData.AccessBoundingBoxes().WorldSpaceBBox();

    unsigned int frame_cnt = inData.FrameCount();
    this->ids.resize(frame_cnt); // allocate data for all frames! We are memory-hungry! but I don't care

    vislib::sys::ConsoleProgressBar cpb;
    cpb.Start("Processing frame data", frame_cnt);

    std::vector<particle_data> pd[2];

    for (unsigned int frame_i = 0; frame_i < frame_cnt; ++frame_i) {
        std::vector<particle_data> &dat_cur = pd[frame_i % 2];
        std::vector<particle_data> &dat_prev = pd[(frame_i + 1) % 2];

        do { // ensure we get the right data
            inData.SetFrameID(frame_i, true);
            if (!inData(0)) return false;
            if (inData.FrameID() != frame_i) vislib::sys::Thread::Sleep(1);
        } while (inData.FrameID() != frame_i);


        if (frame_i > 1) {
            // estimate new particle positions
            //  dat_prev is data from frame_i - 1
            //  dat_cur is data from frame_i - 2 (atm)

            // TODO: Implement

        }

        // ensure data structure for lists
        unsigned int list_cnt = inData.GetParticleListCount();
        if (frame_i > 0) {
            if (ids[frame_i - 1].size() != list_cnt) {
                vislib::sys::Log::DefaultLog.WriteError("Data sets changes lists over time. Unsupported!");
                return false;
            }
        }
        this->ids[frame_i].resize(list_cnt);
        dat_cur.resize(list_cnt);

        for (unsigned int list_i = 0; list_i < list_cnt; ++list_i) {
            // ensure data structure for particls
            auto& parts = inData.AccessParticles(list_i);
            if (frame_i > 0) {
                if (ids[frame_i - 1][list_i].size() != parts.GetCount()) {
                    vislib::sys::Log::DefaultLog.WriteError("Data sets changes particle numbers in list over time. Unsupported!");
                    return false;
                }
            }
            this->ids[frame_i][list_i].resize(parts.GetCount());

            // copy data locally (for being data_prev in the next iteration)
            copyData(dat_cur[list_i], parts);

            // Now compute ids for later sorting
            //  Here ids are the indices of the same particle in the last frame!
            if (frame_i == 0) {
                // initialize with identity for frame 0
                for (unsigned int part_i = 0; part_i < parts.GetCount(); ++part_i) {
                    this->ids[frame_i][list_i][part_i] = part_i;
                }
            } else {
                // minimize sum of distances to particles from last frame

                // TODO: Implement

            }
        }

        inData.Unlock();
        cpb.Set(frame_i);
    }
    cpb.Stop();

    vislib::sys::Log::DefaultLog.WriteInfo("Particle Sorting ID Data updated.");
    return true;
}

void datatools::ParticleSortFixHack::updateData(
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    data.resize(inData.GetParticleListCount());

    for (unsigned int i = 0; i < static_cast<unsigned int>(data.size()); i++) {
        copyData(data[i], inData.AccessParticles(i));

        // TODO: Implement
    }

    outDataTime = inData.FrameID();
}

void datatools::ParticleSortFixHack::copyData(particle_data& tar, core::moldyn::SimpleSphericalParticles& src) {
    tar.parts = src;

    const unsigned char* colPtr = static_cast<const unsigned char*>(src.GetColourData());
    const unsigned char* vertPtr = static_cast<const unsigned char*>(src.GetVertexData());
    unsigned int colSize = 0;
    unsigned int colStride = src.GetColourDataStride();
    unsigned int vertSize = 0;
    unsigned int vertStride = src.GetVertexDataStride();
    bool vertRad = false;

    // colour
    switch (src.GetColourDataType()) {
    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE: break;
    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: colSize = 3; break;
    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: colSize = 4; break;
    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: colSize = 12; break;
    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: colSize = 16; break;
    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: colSize = 4; break;
    default: break;
    }
    if (colStride < colSize) colStride = colSize;

    // radius and position
    switch (src.GetVertexDataType()) {
    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE: break;
    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ: vertSize = 12; break;
    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR: vertSize = 16; break;
        // We do not support short-vertices ATM
    default: break;
    }
    if (vertStride < vertSize) vertStride = vertSize;

    tar.dat.EnforceSize((colSize + vertSize) * tar.parts.GetCount());
    tar.parts.SetVertexData(src.GetVertexDataType(), tar.dat.At(0), colSize + vertSize);
    tar.parts.SetColourData(src.GetColourDataType(), tar.dat.At(vertSize), colSize + vertSize);

    for (UINT64 pi = 0; pi < tar.parts.GetCount(); ++pi) {
        ::memcpy(tar.dat.At(pi * (colSize + vertSize) + 0), vertPtr, vertSize);
        vertPtr += vertStride;
        ::memcpy(tar.dat.At(pi * (colSize + vertSize) + vertSize), colPtr, colSize);
        colPtr += colStride;
    }

}
