/*
 * ParticleDataCall.h
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINCUDAPLUGIN_PARTICLEDATACALL_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_PARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/math/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include <vector>

namespace megamol {
namespace protein_cuda {

    /**
     * Base class of rendering graph calls and of data interfaces for 
     * molecular data (e.g. protein-solvent-systems).
     */

    class ParticleDataCall : public megamol::core::AbstractGetData3DCall {
    public:

        /** Index of the 'GetData' function */
        static const unsigned int CallForGetData;

        /** Index of the 'GetExtent' function */
        static const unsigned int CallForGetExtent;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "ParticleDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get particle data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch( idx) {
                case 0:
                    return "GetData";
                case 1:
                    return "GetExtend";
            }
			return "";
        }

        /** Ctor. */
        ParticleDataCall(void);

        /** Dtor. */
        virtual ~ParticleDataCall(void);

        // -------------------- get and set routines --------------------

        /**
         * Set the number of particles.
         *
         * @param cnt The number of particles.
         */
        void SetParticleCount(unsigned int cnt) { this->particleCount = cnt; }

        /**
         * Get the number of particles.
         *
         * @return The number of particles..
         */
        unsigned int ParticleCount() const { return this->particleCount; }

        /**
         * Assign the particle list.
         *
         * @params partList The particle list.
         */
        void SetParticles(float* partList) { this->particles = partList; }

        /**
         * Access the particle list.
         *
         * @return The particle list.
         */
        const float* Particles() const { return this->particles; }

        /**
         * Assign the color list.
         *
         * @params col The clor list.
         */
        void SetColors(float* col) { this->colors = col; }

        /**
         * Access the color list.
         *
         * @return The color list.
         */
        const float* Colors() const { return this->colors; }

        /**
         * Assign the charges list.
         *
         * @params chargesList The clor list.
         */
        void SetCharges(float* chargesList) { this->charges = chargesList; }

        /**
         * Access the charges list.
         *
         * @return The charges list.
         */
        const float* Charges() const { return this->charges; }

    private:
        // -------------------- variables --------------------

        /** stores the number of particles */
        unsigned int particleCount;

        /** the particle list */
        float* particles;

        /** the color list */
        float* colors;

        /** the charges list */
        float* charges;

    };

    /** Description class typedef */
    typedef megamol::core::CallAutoDescription<ParticleDataCall> ParticleDataCallDescription;


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_PARTICLEDATACALL_H_INCLUDED */
