/*
 * ForceDataCall.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FORCEDATACALL_H_INCLUDED
#define VISLIB_FORCEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "Call.h"
#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"

namespace megamol {
namespace protein {

    /**
     * Base class for transferring force data between renderer and loader.
     */
    class ForceDataCall  : public megamol::core::AbstractGetData3DCall {
    public:

        /** Index of the 'GetData' function */
        static const unsigned int CallForGetForceData;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "ForceDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get force data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 1;
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
                    return "GetForceData";
            }
			return "";
        }

        /** Ctor. */
        ForceDataCall(void);

        /** Dtor. */
        virtual ~ForceDataCall(void);

        
        /**
         * Gets the number of forces being transferred.
         *
         * @return The number of forces being transferred.
         */
        inline unsigned int ForceCount(void) {
            return this->forceCount;
        }
        
        /**
         * Gets the atom id array for the force list.
         *
         * @return Pointer to atom id array (unsigned ints).
         */
        inline const unsigned int* AtomIDs(void) {
            return this->forceAtomIDs;
        }
        
        /**
         * Gets the array of forces.
         *
         * @return Pointer to float array of forces in x1,y1,z1,x2,y2... order.
         */
        inline const float* Forces(void) {
            return this->forceArray;
        }

        /**
         * Sets the new force values.
         *
         * @param count The number of atomIDs for which forces are being provided
         * (not necessarily the total number of atoms).
         * @param atomIDs Pointer to the list of atom ids for which forces are being provided.
         * @param forces Pointer to float array of forces in x1,y1,z1,x2,y2... order. Forces
         * should be provided in the same order as the atom ids.
         */
        void SetForces(unsigned int count, const unsigned int *atomIDs, const float *forces);


    private:

        /** The number of forces being applied */
        unsigned int forceCount;

        /** The array of atom ids that correspond to the forces in the force list */
        const unsigned int *forceAtomIDs;

        /** The array of forces as x,y,z floats */
        const float *forceArray;

    };

    /** Description class typedef */
    typedef megamol::core::CallAutoDescription<ForceDataCall> ForceDataCallDescription;

} /* end namespace protein */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FORCEDATACALL_H_INCLUDED */
