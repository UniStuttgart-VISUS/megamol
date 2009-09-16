/*
 * Camera.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERA_H_INCLUDED
#define VISLIB_CAMERA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameters.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace graphics {


    /**
     * Base class for all camera implementations.
     *
     * This camera base class should not be used directly. There are 
     * specialisations for Direct3D and OpenGL which can apply the current
     * camera transformation directly to the transformation state of the 
     * underlying graphics API.
     *
     * The behaviour of the camera is basically determined by its 
     * CameraParameters. CameraParameters can be shared between different
     * cameras, which behave in the same way then. Sharing is implemented via
     * reference counting. The shared parameters can be overwritten on a 
     * per-camera basis in order to make each camera behave slightly different.
     * Consider the following example: A set of cameras shares the same 
     * camera parameters. They therefore cover the same virtual viewport, which
     * is the complete viewport of a tiled display. Each camera then uses a
     * CameraParamsTileRectOverride to overwrite its tile such that the camera
     * covers only a part of the display. One camera could also specify the 
     * whole virtual viewport as its tile and would therefore generate an 
     * overview image, e. g. for the head node. Overwriting some of the camera
     * parameters works by wrapping the exising parameters of the camera with
     * an *Override class, e. g. somthing like <code>camera.SetParameters(new 
     * CameraParamsTileRectOverride(camera.GetParameters()));</code>.
     *
     * In order to implement the above-mentioned tiled-display pattern, the
     * camera parameters must be communicated to different render nodes. This
     * is not anchored in the Camera/CameraParameters architecture directly, but
     * implemented in the cluster VISlib. The two classes
     * vislib::net::cluster::AbstractControllerNode and 
     * vislib::net::cluster::AbstractControlledNode can be used to distribute
     * camera parameters to a cluster of render nodes.
     *
     * The parameters of a camera can be manipulated by a set of manipulators
     * that translate e. g. mouse motion into a rotation or translation. Such
     * manipluators are e. g. CameraRotate2D or CameraZoom2DAngle. They are 
     * derived from AbstractCursorEvent. Read the respective documentation for
     * more information about cursor events.
     */
    class Camera {

    public:

        /** Ctor. */
        Camera(void);

        /** 
         * Ctor. Initialises the camera with the given camera parameters.
         *
         * @param params The camera parameters to be used. Must not be NULL.
         */
        Camera(const SmartPtr<CameraParameters>& params);

        /**
         * Copy ctor.
         *
         * @param rhs The right hand side operand.
         */
        Camera(const Camera& rhs);

        /** Dtor. */
        virtual ~Camera(void);


        /**
         * Compute the view frustum of the camera in world coordinates.
         *
         * @param outFrustum A frustum object to receive the view frustum.
         *
         * @return 'outFrustum'. The out parameter is returned for convenience.
         *
         * @throws IllegalStateException If the projection type of the camera 
         *                               is not supported. 
         */
        SceneSpaceFrustum& CalcViewFrustum(SceneSpaceFrustum& outFrustum);

        /**
         * Compute the view frustum of the camera in camera coordinates.
         *
         * @param outFrustum A frustum object to receive the view frustum.
         *
         * @return 'outFrustum'. The out parameter is returned for convenience.
         *
         * @throws IllegalStateException If the projection type of the camera 
         *                               is not supported.
         */
        SceneSpaceViewFrustum& CalcViewFrustum(
            SceneSpaceViewFrustum& outFrustum) const;

        /**
         * Answers the parameters object.
         *
         * @return The parameters object.
         */
        SmartPtr<CameraParameters>& Parameters(void);

        /**
         * Answers the parameters object.
         *
         * @return The parameters object.
         */
        const SmartPtr<CameraParameters>& Parameters(void) const;

        /**
         * Sets the parameters object.
         *
         * @param params The new parameters object.
         */
        void SetParameters(const SmartPtr<CameraParameters>& params);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this.
         */
        Camera& operator=(const Camera &rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if 'rhs' and 'this' are equal, 'false' otherwise.
         */
        bool operator==(const Camera &rhs) const;

    protected:

        /**
         * Answer whether an update of attributes derived from the camera 
         * parameters is needed, or not.
         *
         * @return 'true' if an update is needed, 'false' otherwise.
         */
        inline bool needUpdate(void) const {
            return this->syncNumber != this->parameters->SyncNumber();
        }

        /**
         * Clears the need-update flag. This should be called after an update
         * of all attributes derived from the camera parameters was performed.
         */
        inline void markAsUpdated(void) {
            this->syncNumber = this->parameters->SyncNumber();
        }

    private:

        /** the syncronisation number */
        unsigned int syncNumber;

        /** the parameters object of this camera */
        SmartPtr<CameraParameters> parameters;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERA_H_INCLUDED */
