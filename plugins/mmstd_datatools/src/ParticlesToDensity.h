/*
 * ParticlesToDensity.h
 *
 * Copyright (C) 2018 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_DATATOOLS_PARTICLESTODENSITY_H_INCLUDED
#define MMSTD_DATATOOLS_PARTICLESTODENSITY_H_INCLUDED
#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/math/Vector.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include <vector>
#include <map>
#include "mmcore/misc/VolumetricDataCall.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module computing a density volume from particles.
     * One day, hopefully some precise particle-cell intersectors
     * will provide a very accurate result.
     */
class ParticlesToDensity : public megamol::core::Module {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "ParticlesToDensity";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Computes a density volume from particles";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        ParticlesToDensity(void);

        /** Dtor */
        virtual ~ParticlesToDensity(void);

    protected:

        /** Lazy initialization of the module */
        virtual bool create(void);

        /** Resource release */
        virtual void release(void);

    private:

        /**
         * Called when the data is requested by this module
         *
         * @param c The incoming call
         *
         * @return True on success
         */
        bool getDataCallback(megamol::core::Call& c);

        bool dummyCallback(megamol::core::Call& c);
        
        bool createVolumeCPU(megamol::core::moldyn::MultiParticleDataCall* c2);

    /**
         * Called when the extend information is requested by this module
         *
         * @param c The incoming call
         *
         * @return True on success
         */
        bool getExtentCallback(megamol::core::Call& c);

        core::param::ParamSlot aggregatorSlot;

        core::param::ParamSlot xResSlot;
        core::param::ParamSlot yResSlot;
        core::param::ParamSlot zResSlot;

        std::vector<std::vector<float>> vol;

        size_t datahash = 0;
        unsigned int time = 0;
        float maxDens = 0.0f;

        /** The slot providing access to the manipulated data */
        megamol::core::CalleeSlot outDataSlot;

        /** The slot accessing the original data */
        megamol::core::CallerSlot inDataSlot;

        core::misc::VolumetricDataCall::Metadata metadata;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_DATATOOLS_PARTICLESTODENSITY_H_INCLUDED */
