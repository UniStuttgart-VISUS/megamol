/*
 * ParticleInstantiator.h
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLEINSTANTIATOR_H_INCLUDED
#define MEGAMOLCORE_PARTICLEINSTANTIATOR_H_INCLUDED
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global colors of multi particle lists
     */
    class ParticleInstantiator : public AbstractParticleManipulator {
    public:

        static const char *ClassName(void) { return "ParticleInstantiator"; }
        static const char *Description(void) { return "makes instances of particles by xyz repetition"; }
        static bool IsAvailable(void) { return true; }

        ParticleInstantiator(void);
        virtual ~ParticleInstantiator(void);

    protected:

        bool InterfaceIsDirty(void);
        void InterfaceResetDirty();

        /**
         * Manipulates the particle data
         *
         * @remarks the default implementation does not changed the data
         *
         * @param outData The call receiving the manipulated data
         * @param inData The call holding the original data
         *
         * @return True on success
         */
        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData) override;

        virtual bool manipulateExtent(
            core::moldyn::MultiParticleDataCall &outData,
            core::moldyn::MultiParticleDataCall &inData) override;

    private:

        megamol::core::param::ParamSlot numInstancesParam;
        megamol::core::param::ParamSlot instanceOffsetParam;

        SIZE_T hash = -1;
        unsigned int frameID = -1;
        std::vector<std::vector<float>> vertData;
        std::vector<std::vector<uint8_t>> colData;
        std::vector<std::vector<float>> dirData;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEINSTANTIATOR_H_INCLUDED */
