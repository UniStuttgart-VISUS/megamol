/*
 * CameraParamsEyeOverride.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAPARAMSEYEOVERRIDE_H_INCLUDED
#define VISLIB_CAMERAPARAMSEYEOVERRIDE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParamsOverride.h"


namespace vislib {
namespace graphics {


    /**
     * Camera parameter override class overriding the eye.
     */
    class CameraParamsEyeOverride : public CameraParamsOverride {

    public:

        /** Ctor. */
        CameraParamsEyeOverride(void);

        /** 
         * Ctor. 
         *
         * Note: This is not a copy ctor! This create a new object and sets 
         * 'params' as the camera parameter base object.
         *
         * @param params The base 'CameraParameters' object to use.
         */
        CameraParamsEyeOverride(const SmartPtr<CameraParameters>& params);

        /** Dtor. */
        ~CameraParamsEyeOverride(void);

        /**
         * Answer the eye for stereo projections.
         *
         * @return The eye for stereo projections.
         */
        virtual StereoEye Eye(void) const;

        /**
         * Sets the eye for stereo projection.
         *
         * @param eye The eye for stereo projection.
         */
        virtual void SetEye(StereoEye eye);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsEyeOverride& operator=(const CameraParamsEyeOverride& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator==(const CameraParamsEyeOverride& rhs) const;

    private:

        /**
         * Indicates that a new base object is about to be set.
         *
         * @param params The new base object to be set.
         */
        virtual void preBaseSet(const SmartPtr<CameraParameters>& params);

        /**
         * Resets the override.
         */
        virtual void resetOverride(void);

        /** eye for stereo projections */
        StereoEye eye;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMSEYEOVERRIDE_H_INCLUDED */

