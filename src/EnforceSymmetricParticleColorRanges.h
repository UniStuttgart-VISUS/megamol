/*
 * EnforceSymmetricParticleColorRanges.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ENFORCESYMMETRICPARTICLECOLORRANGES_H_INCLUDED
#define MEGAMOLCORE_ENFORCESYMMETRICPARTICLECOLORRANGES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class EnforceSymmetricParticleColorRanges : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "EnforceSymmetricParticleColorRanges";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Changes all color index ranges to be symmetric around zero.";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        EnforceSymmetricParticleColorRanges(void);

        /** Dtor */
        virtual ~EnforceSymmetricParticleColorRanges(void);

    protected:

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
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ENFORCESYMMETRICPARTICLECOLORRANGES_H_INCLUDED */
