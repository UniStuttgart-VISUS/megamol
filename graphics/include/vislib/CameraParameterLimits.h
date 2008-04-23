/*
 * CameraParameterLimits.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAPARAMETERLIMITS_H_INCLUDED
#define VISLIB_CAMERAPARAMETERLIMITS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/graphicstypes.h"
#include "vislib/mathtypes.h"
#include "vislib/Serialisable.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace graphics {


    /**
     * Class of limit values to camera parameters used by CameraParameter
     * objects.
     */
    class CameraParameterLimits : public vislib::Serialisable {
    public:

        /** 
         * Answer the default value object 
         *
         * @return The default value object 
         */ 
        static SmartPtr<CameraParameterLimits>& DefaultLimits(void);

        /** Ctor. */
        CameraParameterLimits(void);

        /** 
         * Copy ctor. 
         *
         * @param rhs The right hand side operand
         */
        CameraParameterLimits(const CameraParameterLimits& rhs);

        /** Dtor. */
        virtual ~CameraParameterLimits(void);

        /**
         * Deserialise the object from 'serialiser'. The caller must ensure that
         * the Serialiser is in an acceptable state to deserialise from.
         *
         * @param serialiser The serialiser to deserialise the object from.
         */
        virtual void Deserialise(Serialiser& serialiser);

        /**
         * Sets the limit values for the aperture angle. The function will
         * fail if 'minValue' or 'maxValue' are less or equal zero, or greater
         * or equal PI (180°) or if 'maxValue' is less than 'minValue'.
         *
         * @param minValue The minimum value valid for a (full) aperture angle
         *                 in radians.
         * @param maxValue The maximum value valid for a (full) aperture angle
         *                 in radians.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool LimitApertureAngle(math::AngleRad minValue, math::AngleRad maxValue);

        /**
         * Sets the limit values for the clipping distances. The function will
         * fail if 'minClipDist' is less or equal zero.
         *
         * @param minNearDist The minimum value valid for the near clipping
         *                    plane distance.
         * @param minClipDist The minimum value valid for the distance between
         *                    the near and the far clipping plane.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool LimitClippingDistances(SceneSpaceType minNearDist, 
            SceneSpaceType minClipDist);
        
        /**
         * Sets the limiting minimum value for the focal distance.
         *
         * @param minFocalDist The minimum value valid for the focal distance.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool LimitFocalDistance(SceneSpaceType minFocalDist);

        /**
         * Sets the limiting minimum value for the distance between the 
         * position and the look-at-point of the camera. The function fails if
         * 'minLookAtDist' is less or equal zero.
         *
         * @param minLookAtDist The minimum value valid for the distance 
         *                      between position and look-at-point of a camera.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool LimitLootAtDistance(SceneSpaceType minLookAtDist);

        /** 
         * Answer the maximal aperture angle 
         *
         * @return The maximal aperture angle 
         */
        inline math::AngleRad MaxApertureAngle(void) const {
            return this->maxHalfApertureAngle * 2.0f;
        }

        /** 
         * Answer the maximal half aperture angle 
         *
         * @return The maximal half aperture angle 
         */
        inline math::AngleRad MaxHalfApertureAngle(void) const {
            return this->maxHalfApertureAngle;
        }

        /** 
         * Answer the minimal difference between near and far clipping distance
         *
         * @return The minimal difference between the clipping distance
         */
        inline SceneSpaceType MinClipPlaneDist(void) const {
            return this->minClipPlaneDist;
        } 

        /** 
         * Answer the minimal focal distance 
         *
         * @return The minimal focal distance 
         */
        inline SceneSpaceType MinFocalDist(void) const {
            return this->minFocalDist;
        }

        /** 
         * Answer the minimal aperture angle 
         *
         * @return The minimal aperture angle 
         */
        inline math::AngleRad MinApertureAngle(void) const {
            return this->minHalfApertureAngle * 2.0f;
        }

        /** 
         * Answer the minimal half aperture angle 
         *
         * @return The minimal half aperture angle 
         */
        inline math::AngleRad MinHalfApertureAngle(void) const {
            return this->minHalfApertureAngle;
        }

        /**
         * Answer the minimal distance between position and look at 
         *
         * @return The minimal distance between position and look at
         */
        inline SceneSpaceType MinLookAtDist(void) const {
            return this->minLookAtDist;
        }

        /** 
         * Answer the minimal near clipping distance 
         *
         * @return The minimal near clipping distance 
         */
        inline SceneSpaceType MinNearClipDist(void) const {
            return this->minNearClipDist;
        }

        /** Resets all values to the default values. */
        void Reset(void);

        /**
         * Serialise the object to 'serialiser'. The caller must ensure that
         * the Serialiser is in an acceptable state to serialise to.
         *
         * @param serialiser The serialiser to serialise the object to.
         */
        virtual void Serialise(Serialiser& serialiser) const;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return Reference to this object
         */
        CameraParameterLimits& operator=(const CameraParameterLimits& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if all members are equal, 'false' otherwise.
         */
        bool operator==(const CameraParameterLimits& rhs) const;

    private:

        /** the maximal half aperture angle */
        math::AngleRad maxHalfApertureAngle;

        /** the minimal difference between near and far clipping distance */
        SceneSpaceType minClipPlaneDist;

        /** the minimal focal distance */
        SceneSpaceType minFocalDist;

        /** the minimal half aperture angle */
        math::AngleRad minHalfApertureAngle;

        /** the minimal distance between position and look at */
        SceneSpaceType minLookAtDist;

        /** the minimal near clipping distance */
        SceneSpaceType minNearClipDist;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMETERLIMITS_H_INCLUDED */

