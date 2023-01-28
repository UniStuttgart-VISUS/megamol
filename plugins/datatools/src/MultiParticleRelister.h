/*
 * MultiParticleRelister.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED
#define MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "geometry_calls/ParticleRelistCall.h"
#include "mmcore/CallerSlot.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include <cstdint>
#include <vector>

namespace megamol::datatools {

class MultiParticleRelister : public AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "MultiParticleRelister";
    }
    static const char* Description() {
        return "Reorganizes MultiParticle Lists";
    }
    static bool IsAvailable() {
        return true;
    }

    MultiParticleRelister();
    ~MultiParticleRelister() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    void copyData(const geocalls::SimpleSphericalParticles& inData, const geocalls::ParticleRelistCall& relist);

    core::CallerSlot getRelistInfoSlot;

    size_t inDataHash;
    unsigned int inFrameId;
    size_t inRelistHash;
    size_t outRelistHash;

    geocalls::SimpleSphericalParticles::ColourDataType colDatTyp;
    vislib::graphics::ColourRGBAu8 globCol;
    geocalls::SimpleSphericalParticles::VertexDataType verDatTyp;
    float globRad;
    size_t partSize;
    size_t colOffset;
    std::vector<std::vector<uint8_t>> data;
};

} // namespace megamol::datatools

#endif /* MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED */
