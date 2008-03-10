/*
 * AbstractControllerNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCONTROLLERNODE_H_INCLUDED
#define VISLIB_ABSTRACTCONTROLLERNODE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Socket.h"  // Must be first
#include "vislib/CameraParameterObserver.h"
#include "vislib/ObservableCameraParams.h"
#include "vislib/types.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * By inheriting from this class, it is possible to make a cluster node the 
     * controlling node that propagates the camera settings to all other nodes
     * in the cluster application.
     */
    class AbstractControllerNode : public graphics::CameraParameterObserver {

    public:

        /** Dtor. */
        ~AbstractControllerNode(void);

        /**
         * Begin a batch interaction that accumulates all changes to the camera
         * parameters instead of transferring it directly. All change events 
         * that the object receives until a call to EndBatchInteraction will be
         * accumulated and not directly transferred to the client nodes.
         */
        virtual void BeginBatchInteraction(void);

        /**
         * Ends a batch interaction and transfers all changes to the client 
         * nodes.
         */
        virtual void EndBatchInteraction(void);

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
        virtual void OnEyeChanged(
            const graphics::CameraParameters::StereoEye newValue);

        /**
         * This method is called if the far clipping plane changed.
         *
         * @param newValue The new far clipping plane.
         */
        virtual void OnFarClipChanged(const graphics::SceneSpaceType newValue);

        /**
         * This method is called if the focal distance changed.
         *
         * @param newValue The new forcal distance.
         */
        virtual void OnFocalDistanceChanged(
            const graphics::SceneSpaceType newValue);

        //virtual void OnLimitsChanged(const SmartPtr<CameraParameterLimits>& limits) = 0;

        /**
         * This method is called if the look at point changed.
         *
         * @param newValue The new look at point.
         */
        virtual void OnLookAtChanged(
            const graphics::SceneSpacePoint3D& newValue);

        /**
         * This method is called if the near clipping plane changed.
         *
         * @param newValue The new near clipping plane.
         */
        virtual void OnNearClipChanged(const graphics::SceneSpaceType newValue);

        /**
         * This method is called if the camera position changed.
         *
         * @param newValue The new camera position.
         */
        virtual void OnPositionChanged(
            const graphics::SceneSpacePoint3D& newValue);

        /**
         * This method is called if the projection type changed.
         *
         * @param newValue The new projection type.
         */
        virtual void OnProjectionChanged(
            const graphics::CameraParameters::ProjectionType newValue);

        /**
         * This method is called if the stereo disparity changed.
         *
         * @param newValue The new stereo disparity.
         */
        virtual void OnStereoDisparityChanged(
            const graphics::SceneSpaceType newValue);

        /**
         * This method is called if the screen tile changed.
         *
         * @param newValue The new screen tile.
         */
        virtual void OnTileRectChanged(
            const graphics::ImageSpaceRectangle& newValue);

        /**
         * This method is called if the camera up vector changed.
         *
         * @param newValue The new camera up vector.
         */
        virtual void OnUpChanged(
            const graphics::SceneSpaceVector3D& newValue);

        /**
         * This method is called if the virtual screen size changed.
         *
         * @param newValue The new virtual screen size.
         */
        virtual void OnVirtualViewSizeChanged(
            const graphics::ImageSpaceDimension& newValue);

    protected:

        /** Ctor. */
        AbstractControllerNode(void);

    private:

    };
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCONTROLLERNODE_H_INCLUDED */

