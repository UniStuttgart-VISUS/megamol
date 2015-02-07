/*
 * DirectionalParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "MultiParticleDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Class holding all data of a single directed particle type
     */
    class MEGAMOLCORE_API DirectionalParticles
        : public SimpleSphericalParticles {
    public:

        /** possible values for the direction data */
        enum DirDataType {
            DIRDATA_NONE,
            DIRDATA_FLOAT_XYZ
        };

        /**
         * Ctor
         */
        DirectionalParticles(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        DirectionalParticles(const DirectionalParticles& src);

        /**
         * Dtor
         */
        ~DirectionalParticles(void);

        /**
         * Answer the direction data type
         *
         * @return The direction data type
         */
        inline DirDataType GetDirDataType(void) const {
            return this->dirDataType;
        }

        /**
         * Answer the direction data pointer
         *
         * @return The direction data pointer
         */
        inline const void * GetDirData(void) const {
            return this->dirPtr;
        }

        /**
         * Answer the direction data stride
         *
         * @return The direction data stride
         */
        inline unsigned int GetDirDataStride(void) const {
            return this->dirStride;
        }

        /**
         * Sets the direction data
         *
         * @param t The type of the direction data
         * @param p The pointer to the direction data (must not be NULL if t
         *          is not 'DIRDATA_NONE'
         * @param s The stride of the direction data
         */
        void SetDirData(DirDataType t, const void *p, unsigned int s = 0) {
            ASSERT((p != NULL) || (t == DIRDATA_NONE));
            this->dirDataType = t;
            this->dirPtr = p;
            this->dirStride = s;
        }

        /**
         * Sets the number of objects stored and resets all data pointers!
         *
         * @param cnt The number of stored objects
         */
        void SetCount(UINT64 cnt) {
            this->dirDataType = DIRDATA_NONE;
            this->dirPtr = NULL; // DO NOT DELETE
            SimpleSphericalParticles::SetCount(cnt);
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        DirectionalParticles& operator=(const DirectionalParticles& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const DirectionalParticles& rhs) const;

    private:

        /** The direction data type */
        DirDataType dirDataType;

        /** The direction data pointer */
        const void *dirPtr;

        /** The direction data stride */
        unsigned int dirStride;

    };


    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<DirectionalParticles>;


    /**
     * Call for multi-stream particle data.
     */
    class MEGAMOLCORE_API DirectionalParticleDataCall
        : public AbstractParticleDataCall<DirectionalParticles> {
    public:

        /** typedef for legacy name */
        typedef DirectionalParticles Particles;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "DirectionalParticleDataCall";
        }

        /** Ctor. */
        DirectionalParticleDataCall(void);

        /** Dtor. */
        virtual ~DirectionalParticleDataCall(void);

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        DirectionalParticleDataCall& operator=(const DirectionalParticleDataCall& rhs);

    };


    /** Description class typedef */
    typedef factories::CallAutoDescription<DirectionalParticleDataCall>
        DirectionalParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED */
