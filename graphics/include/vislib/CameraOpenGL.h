/*
 * CameraOpenGL.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAOPENGL_H_INCLUDED
#define VISLIB_CAMERAOPENGL_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


/** TODO: DEBUG: REMOVE THE FOLLOWING LINES */
#ifndef VISLIB_ENABLE_OPENGL
#define VISLIB_ENABLE_OPENGL 1
#endif


/**
 * To use this class you must define VISLIB_ENABLE_OPENGL.
 * Additionally you may need to link against several third party libraries:
 *  GL GLU
 */
#ifdef VISLIB_ENABLE_OPENGL


#include <GL/glu.h>

#include "vislib/Camera.h"
#include "vislib/Point3D.h"
#include "vislib/Vector3D.h"


namespace vislib {
namespace graphics {

    /**
     * camera class for openGL applications
     */
    class CameraOpenGL: public Camera {
    public:
        /**
         * default ctor
         */
        CameraOpenGL(void) : Camera(), viewNeedsUpdate(true), projNeedsUpdate(true) {
        }

        /**
         * copy ctor
         *
         * @param rhs CameraOpenGL object to copy from.
         */
        CameraOpenGL(const CameraOpenGL& rhs) : Camera(rhs), viewNeedsUpdate(true), projNeedsUpdate(true) {
        }

        /**
         * Initialization ctor
         *
         * @param beholder The beholder the new camera object will be 
         *                 associated with.
         */
        template <class Tp > CameraOpenGL(const Beholder<Tp>& beholder) 
            : Camera(rhs), viewNeedsUpdate(true), projNeedsUpdate(true) {
        }

        /**
         * dtor
         */
        virtual ~CameraOpenGL(void) {
        }

        /**
         * Multipiles the current openGL matrix with the projection matrix of 
         * the frustum of this camera.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void glMultProjectionMatrix(void) {
            if (this->NeedUpdate()) {
                this->viewNeedsUpdate = this->projNeedsUpdate = true;
                this->ClearUpdateFlaggs();
            }
            
            if (this->projNeedsUpdate) {
                Camera::CalcFrustumParameters(left, right, bottom, top, nearClip, farClip);
                this->projNeedsUpdate = false;
            }

            if (this->GetProjectionType() != Camera::MONO_ORTHOGRAPHIC) {
                ::glFrustum(left, right, bottom, top, nearClip, farClip);
            } else {
                ::glOrtho(left, right, bottom, top, nearClip, farClip);
            }
        }

        /**
         * Multiplies the current openGL matrix with the viewing matrix of this
         * camera.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void glMultViewMatrix(void) {
            if (this->NeedUpdate()) {
                this->viewNeedsUpdate = this->projNeedsUpdate = true;
                this->ClearUpdateFlaggs();
            }

            if (this->viewNeedsUpdate) {
                Camera::CalcViewParameters(pos, lookDir, up);
                this->viewNeedsUpdate = false;
            }

            ::gluLookAt(pos.X(), pos.Y(), pos.Z(), 
                pos.X() + lookDir.X(), pos.Y() + lookDir.Y(), pos.Z() + lookDir.Z(), 
                up.X(), up.Y(), up.Z());
        }

        /**
         * Assignment operator
         */
        CameraOpenGL& operator=(const CameraOpenGL& rhs) {
            Camera::operator=(rhs);
            return *this;
        }

    private:

        /** Flaggs which members need updates */
        bool viewNeedsUpdate, projNeedsUpdate;

        /** viewing frustum minimal x */
        Camera::SceneSpaceValue left;

        /** viewing frustum maximal x */
        Camera::SceneSpaceValue right;

        /** viewing frustum minimal y */
        Camera::SceneSpaceValue bottom;

        /** viewing frustum maximal y */
        Camera::SceneSpaceValue top;

        /** viewing frustum minimal z */
        Camera::SceneSpaceValue nearClip;

        /** viewing frustum maximal z */
        Camera::SceneSpaceValue farClip;

        /** projection viewer position */
        math::Point3D<Camera::SceneSpaceValue> pos;

        /** projection viewing direction */ 
        math::Vector3D<Camera::SceneSpaceValue> lookDir;

        /** projection up vector */
        math::Vector3D<Camera::SceneSpaceValue> up;

    };

} /* end namespace graphics */
} /* end namespace vislib */


#endif /* VISLIB_ENABLE_OPENGL */

#endif /* VISLIB_CAMERAOPENGL_H_INCLUDED */
