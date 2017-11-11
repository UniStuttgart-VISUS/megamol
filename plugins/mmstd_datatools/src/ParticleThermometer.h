/*
 * ParticleThermometer.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED
#define MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED
#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/Module.h"
#include <vector>


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class ParticleThermometer : public megamol::core::Module {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "ParticleThermometer";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Uses ";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        ParticleThermometer(void);

        /** Dtor */
        virtual ~ParticleThermometer(void);

    protected:

        /** Lazy initialization of the module */
        virtual bool create(void);

        /** Resource release */
        virtual void release(void);

    private:

        core::param::ParamSlot cyclXSlot;
        core::param::ParamSlot cyclYSlot;
        core::param::ParamSlot cyclZSlot;
        size_t datahash;
        unsigned int time;
        std::vector<float> newColors;
        float minCol, maxCol;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED */
