/*
 * CameraParameterObserver.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAPARAMETEROBSERVER_H_INCLUDED
#define VISLIB_CAMERAPARAMETEROBSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameters.h"


namespace vislib {
namespace graphics {


    /**
     * Classes that inherit from this one can be informed about camera parameter
     * changes if they register to a parameter class.
     */
    class CameraParameterObserver {

    public:

        /** Dtor. */
        virtual ~CameraParameterObserver(void);

        /**
         * This method is called if the aperture angle changed.
         *
         * @param newValue The new aperture angle.
         */
        virtual void OnApertureAngleChanged(const math::AngleDeg newValue);

        /**
         * This method is called if the stereo eye changed.
         *
         * @param newValue The new stereo eye.
         */
        virtual void OnEyeChanged(const CameraParameters::StereoEye newValue);

        /**
         * This method is called if the far clipping plane changed.
         *
         * @param newValue The new far clipping plane.
         */
        virtual void OnFarClipChanged(const SceneSpaceType newValue);

        /**
         * This method is called if the focal distance changed.
         *
         * @param newValue The new forcal distance.
         */
        virtual void OnFocalDistanceChanged(const SceneSpaceType newValue);

        //virtual void OnLimitsChanged(const SmartPtr<CameraParameterLimits>& limits) = 0;

        /**
         * This method is called if the look at point changed.
         *
         * @param newValue The new look at point.
         */
        virtual void OnLookAtChanged(const SceneSpacePoint3D& newValue);

        /**
         * This method is called if the near clipping plane changed.
         *
         * @param newValue The new near clipping plane.
         */
        virtual void OnNearClipChanged(const SceneSpaceType newValue);

        /**
         * This method is called if the camera position changed.
         *
         * @param newValue The new camera position.
         */
        virtual void OnPositionChanged(const SceneSpacePoint3D& newValue);

        /**
         * This method is called if the projection type changed.
         *
         * @param newValue The new projection type.
         */
        virtual void OnProjectionChanged(
            const CameraParameters::ProjectionType newValue);

        /**
         * This method is called if the stereo disparity changed.
         *
         * @param newValue The new stereo disparity.
         */
        virtual void OnStereoDisparityChanged(const SceneSpaceType newValue);

        /**
         * This method is called if the screen tile changed.
         *
         * @param newValue The new screen tile.
         */
        virtual void OnTileRectChanged(const ImageSpaceRectangle& newValue);

        /**
         * This method is called if the camera up vector changed.
         *
         * @param newValue The new camera up vector.
         */
        virtual void OnUpChanged(const SceneSpaceVector3D& newValue);

        /**
         * This method is called if the virtual screen size changed.
         *
         * @param newValue The new virtual screen size.
         */
        virtual void OnVirtualViewSizeChanged(
            const ImageSpaceDimension& newValue);

    protected:

        /** Ctor. */
        CameraParameterObserver(void);

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMETEROBSERVER_H_INCLUDED */

