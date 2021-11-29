/*
 * MultiParticleRelister.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED
#define MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools/AbstractParticleManipulator.h"
#include "geometry_calls/ParticleRelistCall.h"
#include "mmcore/CallerSlot.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include <cstdint>
#include <vector>

namespace megamol {
namespace datatools {

class MultiParticleRelister : public AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "MultiParticleRelister";
    }
    static const char* Description(void) {
        return "Reorganizes MultiParticle Lists";
    }
    static bool IsAvailable(void) {
        return true;
    }

    MultiParticleRelister();
    virtual ~MultiParticleRelister();

protected:
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

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

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED */
