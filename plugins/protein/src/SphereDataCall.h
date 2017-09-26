/*
 * SphereDataCall.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SPHEREDATACALL_H_INCLUDED
#define MMPROTEINPLUGIN_SPHEREDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/math/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include <vector>

namespace megamol {
namespace protein {

    /**
     * Base class of rendering graph calls and of data interfaces for 
     * sphere data (e.g. coarse grain protein-solvent-systems).
     *
     * Note that all data has to be sorted!
     * There are no IDs anymore, always use the index values into the right 
     * tables.
     *
     * All data has to be stored in the corresponding data source object. This
     * interface object will only pass pointers to the renderer objects. A
     * compatible data source should therefore use the public nested class of
     */

	class SphereDataCall : public megamol::core::AbstractGetData3DCall {
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
            return "SphereDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get sphere data";
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
        SphereDataCall(void);

        /** Dtor. */
        virtual ~SphereDataCall(void);

        // -------------------- get and set routines --------------------

        /**
         * Get the total number of spheres.
         *
         * @return The sphere count.
         */
        unsigned int SphereCount(void) const { return sphereCount; }

        /**
         * Get the spheres (position + radius).
         *
         * @return The sphere array.
         */
        const float* Spheres(void) const { return spheres; }

        /**
         * Get the sphere colors.
         *
         * @return The sphere colors array.
         */
        const unsigned char* SphereColors(void) const { return colors; }

        /**
         * Get the sphere charge.
         *
         * @return The sphere charge array.
         */
        const float* SphereCharges(void) const { return charges; }

        /**
         * Get the minimum charge.
         *
         * @return The minimum charge.
         */
        float MinimumCharge(void) const { return this->minCharge; }

        /**
         * Get the maximum charge.
         *
         * @return The maximum charge.
         */
        float MaximumCharge(void) const { return this->maxCharge; }

        /**
         * Set the spheres.
         * 
         * @param sphereCnt The number of spheres.
         * @param data      The pointer to the sphere data array.
         * @param type      The pointer to the type array.
         * @param charge    The pointer to the charge array.
         * @param color     The pointer to the color array.
         */
        void SetSpheres( unsigned int sphereCnt, float* data, 
            unsigned int* type, float* charge, unsigned char* color);

        /**
         * Set charge range.
         *
         * @param min The minimum charge.
         * @param max The maximum charge.
         */
        void SetChargeRange( float min, float max) { 
            this->minCharge = min;
            this->maxCharge = max;
        }

    private:
        // -------------------- variables --------------------

        /** The number of spheres. */
        unsigned int sphereCount;
        /** The array of spheres (position + radius). */
        float* spheres;
        /** The array of sphere colors */
        unsigned char* colors;
        /** The array of sphere charges */
        float* charges;
        /** The minimum charge */
        float minCharge;
        /** The maximum charge */
        float maxCharge;
        /** The array of sphere types */
        unsigned int* types;
        
    };

    /** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<SphereDataCall> SphereDataCallDescription;


} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_SPHEREDATACALL_H_INCLUDED */
