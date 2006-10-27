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


#include <GL/glu.h>

#include "vislib/graphicstypes.h"
#include "vislib/Camera.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"


namespace vislib {
namespace graphics {
namespace gl {

    /**
     * camera class for openGL applications
     */
    class CameraOpenGL: public vislib::graphics::Camera {
    public:
        /**
         * default ctor
         */
        CameraOpenGL(void);

        /**
         * copy ctor
         *
         * @param rhs CameraOpenGL object to copy from.
         */
        CameraOpenGL(const CameraOpenGL& rhs);

        /**
         * Initialization ctor
         *
         * @param beholder The beholder the new camera object will be 
         *                 associated with.
         */
        CameraOpenGL(Beholder* beholder);

        /**
         * dtor
         */
        virtual ~CameraOpenGL(void);

        /**
         * Multipiles the current openGL matrix with the projection matrix of 
         * the frustum of this camera.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void glMultProjectionMatrix(void);

        /**
         * Multiplies the current openGL matrix with the viewing matrix of this
         * camera.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void glMultViewMatrix(void);

        /**
         * Assignment operator
         */
        CameraOpenGL& operator=(const CameraOpenGL& rhs);

    private:

        /** Flaggs which members need updates */
        bool viewNeedsUpdate, projNeedsUpdate;

        /** viewing frustum minimal x */
        SceneSpaceType left;

        /** viewing frustum maximal x */
        SceneSpaceType right;

        /** viewing frustum minimal y */
        SceneSpaceType bottom;

        /** viewing frustum maximal y */
        SceneSpaceType top;

        /** viewing frustum minimal z */
        SceneSpaceType nearClip;

        /** viewing frustum maximal z */
        SceneSpaceType farClip;

        /** projection viewer position */
        math::Point<SceneSpaceType, 3> pos;

        /** projection viewing direction */ 
        math::Vector<SceneSpaceType, 3> lookDir;

        /** projection up vector */
        math::Vector<SceneSpaceType, 3> up;

    };

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */


#endif /* VISLIB_CAMERAOPENGL_H_INCLUDED */
