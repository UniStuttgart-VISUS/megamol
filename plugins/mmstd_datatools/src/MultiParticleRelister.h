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

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/CallerSlot.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include <vector>
#include <cstdint>
#include "mmcore/moldyn/ParticleRelistCall.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    class MultiParticleRelister : public AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "MultiParticleRelister"; }
        static const char *Description(void) { return "Reorganizes MultiParticle Lists"; }
        static bool IsAvailable(void) { return true; }

        MultiParticleRelister();
        virtual ~MultiParticleRelister();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        void copyData(const core::moldyn::SimpleSphericalParticles& inData, const core::moldyn::ParticleRelistCall& relist);

        core::CallerSlot getRelistInfoSlot;

        size_t inDataHash;
        unsigned int inFrameId;
        size_t inRelistHash;
        size_t outRelistHash;

        core::moldyn::SimpleSphericalParticles::ColourDataType colDatTyp;
        vislib::graphics::ColourRGBAu8 globCol;
        core::moldyn::SimpleSphericalParticles::VertexDataType verDatTyp;
        float globRad;
        size_t partSize;
        size_t colOffset;
        std::vector<std::vector<uint8_t> > data;


    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_STD_DATATOOLS_MULTIPARTICLERELISTER_H_INCLUDED */
