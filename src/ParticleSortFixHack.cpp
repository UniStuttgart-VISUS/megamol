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

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleSortFixHack::ParticleSortFixHack
 */
datatools::ParticleSortFixHack::ParticleSortFixHack(void)
        : AbstractParticleManipulator("outData", "indata"), data(),
        inDataHash(-1), inDataTime(-1), outDataHash(0), outDataTime(0) {
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

    // update internal data if required
    if ((inData.DataHash() == 0) || (inData.DataHash() != inDataHash)
            || (inData.FrameID() != inDataTime)
            || (data.size() != inData.GetParticleListCount())) {
        inDataHash = inData.DataHash();
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

void datatools::ParticleSortFixHack::updateData(
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    data.resize(inData.GetParticleListCount());

    for (unsigned int i = 0; i < static_cast<unsigned int>(data.size()); i++) {
        copyData(data[i], inData.AccessParticles(i));

        // TODO: Implement
    }

    outDataHash++;
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
