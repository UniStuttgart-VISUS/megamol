/*
 * CameraParameterObservable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAPARAMETEROBSERVABLE_H_INCLUDED
#define VISLIB_CAMERAPARAMETEROBSERVABLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameterObserver.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {
namespace graphics {


    /**
     * This class implements the interface for registering 
     * CameraParameterObservers. This whole process of registering and
     * removing observers as well as convenience methods for firing the events
     * are implemented here.
     */
    class CameraParameterObservable {

    public:

        /** Dtor. */
        ~CameraParameterObservable(void);

        /**
         * Register a new observer that is informed about camera parameter 
         * changes.
         *
         * NOTE: The object does NOT ownership of the object designated by
         * 'observer'. The caller is responsible that 'observer' exists as
         * long as it is registered.
         *
         * @param observer The observer to be registered. This must not be
         *                 NULL.
         */
        virtual void AddCameraParameterObserver(
            CameraParameterObserver *observer);

        /**
         * Removes the observer 'observer' from the list ob registered 
         * camera parameter observers. It is safe to remove non-registered
         * observers.
         *
         * @param observer The observer to be removed. This must not be NULL.
         */
        virtual void RemoveCameraParameterObserver(
            CameraParameterObserver *observer);

    protected:

        /** Ctor. */
        CameraParameterObservable(void);

        /**
         * Inform all registered observers about a change of the aperture
         * angle.
         *
         * @param newValue The new aperture angle.
         */
        void fireApertureAngleChanged(const math::AngleDeg newValue);

        /**
         * Inform all registered observers about a change of the stereo eye.
         *
         * @param newValue The new stereo eye.
         */
        void fireEyeChanged(const CameraParameters::StereoEye newValue);

        /**
         * Inform all registered observers about a change of the far clipping
         * plane.
         *
         * @param newValue The new far clipping plane.
         */
        void fireFarClipChanged(const SceneSpaceType newValue);

        /**
         * Inform all registered observers about a change of the focal distance.
         *
         * @param newValue The new focal distance.
         */
        void fireFocalDistanceChanged(const SceneSpaceType newValue);

        /**
         * Inform all registered observers about a change of the look at point.
         *
         * @param newValue The new look at point.
         */
        void fireLookAtChanged(const SceneSpacePoint3D& newValue);

        /**
         * Inform all registered observers about a change of the near clipping
         * plane.
         *
         * @param newValue The new nwar clipping plane.
         */
        void fireNearClipChanged(const SceneSpaceType newValue);

        /**
         * Inform all registered observers about a change of the camera
         * position.
         *
         * @param newValue The new camera position.
         */
        void firePositionChanged(const SceneSpacePoint3D& newValue);

        /**
         * Inform all registered observers about a change of the type of 
         * projection.
         *
         * @param newValue The new type of projection.
         */
        void fireProjectionChanged(
            const CameraParameters::ProjectionType newValue);

        /**
         * Inform all registered observers about a change of the stereo 
         * disparity.
         *
         * @param newValue The new stereo disparity.
         */
        void fireStereoDisparityChanged(const SceneSpaceType newValue);

        /**
         * Inform all registered observers about a change of the image tile.
         *
         * @param newValue The new image tile.
         */
        void fireTileRectChanged(const ImageSpaceRectangle& newValue);

        /**
         * Inform all registered observers about a change of the camera up 
         * vector.
         *
         * @param newValue The new camera up vector.
         */
        void fireUpChanged(const SceneSpaceVector3D& newValue);

        /**
         * Inform all registered observers about a change of the virtual screen
         * size.
         *
         * @param newValue The new virtual screen size.
         */
        void fireVirtualViewSizeChanged(const ImageSpaceDimension& newValue);

    private:

        /** The list of registered CameraParameterObservers. */
        SingleLinkedList<CameraParameterObserver *> camParamObservers;


    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMETEROBSERVABLE_H_INCLUDED */

