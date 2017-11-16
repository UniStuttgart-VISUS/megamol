/*
 * DirectionalParticleDataCall.h
 *
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_MOLDYN_ELLIPSOIDDATACALL_H_INCLUDED
#define MMSTD_MOLDYN_ELLIPSOIDDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmstd_moldyn.h"

#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <cassert>


namespace megamol {
namespace stdplugin {
namespace moldyn {

    /**
     * Class holding all data of a single ellipsoidal particle type
     *
     * Consistency Assumptions:
     *  Position-Data must be VERTDATA_FLOAT_XYZ
     *  The global radius is never used
     *  You must set the quaternion data
     *  You must set the radii data
     */
    class MMSTD_MOLDYN_API EllipsoidalParticles : public core::moldyn::SimpleSphericalParticles {
    public:

        /**
         * Ctor
         */
        EllipsoidalParticles(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        EllipsoidalParticles(const EllipsoidalParticles& src);

        /**
         * Dtor
         */
        ~EllipsoidalParticles(void);

        /**
         * Answer the quaternion data pointer
         *
         * @return The quaternion data pointer
         */
        inline const float * GetQuatData(void) const {
            return this->quatPtr;
        }


        /**
        * Answer the quaternion data stride
        *
        * @return The quaternion data stride
        */
        inline unsigned int GetQuatDataStride(void) const {
            return this->quatStride;
        }


        /**
        * Answer the radii data pointer
        *
        * @return The radii data pointer
        */
        inline const float * GetRadiiData(void) const {
            return this->radPtr;
        }

        
        /**
         * Answer the radii data stride
         *
         * @return The radii data stride
         */
        inline unsigned int GetRadiiDataStride(void) const {
            return this->radStride;
        }

        /**
         * Sets the quaternion data
         *
         * @param p The pointer to the quaternion data (must not be NULL)
         * @param s The stride of the direction data
         */
        void SetQuatData(const float *p, unsigned int s = 0) {
            assert(p != nullptr);
            this->quatPtr = p;
            this->quatStride = s;
        }


        /**
        * Sets the radii data
        *
        * @param p The pointer to the radii data (must not be NULL)
        * @param s The stride of the radii data
        */
        void SetRadData(const float *p, unsigned int s = 0) {
            assert(p != nullptr);
            this->radPtr = p;
            this->radStride = s;
        }


        /**
         * Sets the number of objects stored and resets all data pointers!
         *
         * @param cnt The number of stored objects
         */
        void SetCount(UINT64 cnt) {
            this->quatPtr = nullptr; // DO NOT DELETE
            this->radPtr = nullptr; // DO NOT DELETE
            SimpleSphericalParticles::SetCount(cnt);
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        EllipsoidalParticles& operator=(const EllipsoidalParticles& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const EllipsoidalParticles& rhs) const;

    private:

        /** The quaternion data pointer */
        const float *quatPtr;

        /** The quaternion data stride */
        unsigned int quatStride;

        /** The radii data pointer */
        const float *radPtr;

        /** The radii data stride */
        unsigned int radStride;
    };


    MEGAMOLCORE_APIEXT template class MMSTD_MOLDYN_API core::moldyn::AbstractParticleDataCall<EllipsoidalParticles>;


    /**
     * Call for multi-stream particle data.
     */
    class MMSTD_MOLDYN_API EllipsoidalParticleDataCall : public core::moldyn::AbstractParticleDataCall<EllipsoidalParticles> {
    public:

        /** typedef for legacy name */
        typedef EllipsoidalParticles Particles;

        static const unsigned int CallForGetData;
        static const unsigned int CallForGetExtents;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "EllipsoidalParticleDataCall";
        }

        /** Ctor. */
        EllipsoidalParticleDataCall(void);

        /** Dtor. */
        virtual ~EllipsoidalParticleDataCall(void);

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        EllipsoidalParticleDataCall& operator=(const EllipsoidalParticleDataCall& rhs);

    };


    /** Description class typedef */
    typedef core::factories::CallAutoDescription<EllipsoidalParticleDataCall>
        EllipsoidalParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_MOLDYN_ELLIPSOIDDATACALL_H_INCLUDED */
