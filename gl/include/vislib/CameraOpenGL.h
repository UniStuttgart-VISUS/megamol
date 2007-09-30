/*
 * CameraOpenGL.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAOPENGL_H_INCLUDED
#define VISLIB_CAMERAOPENGL_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Camera.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Class of the openGL implementation of 'Camera'
     */
    class CameraOpenGL : public Camera {
    public:

        /** Ctor. */
        CameraOpenGL(void);

        /** 
         * Ctor. Initialises the camera with the given camera parameters.
         *
         * @param params The camera parameters to be used. Must not be NULL.
         */
        CameraOpenGL(const SmartPtr<CameraParameters>& params);

        /**
         * Copy ctor.
         *
         * @param rhs The right hand side operand.
         */
        CameraOpenGL(const Camera& rhs);

        /** Dtor. */
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
         * Answer the projection matrix of the frustum of this camera.
         *
         * @param mat Points to an array of 16 floats receiving the matrix.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void ProjectionMatrix(float *mat);

        /**
         * Answer the view matrix of the frustum of this camera.
         *
         * @param mat Points to an array of 16 floats receiving the matrix.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void ViewMatrix(float *mat);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this.
         */
        CameraOpenGL& operator=(const Camera &rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if 'rhs' and 'this' are equal, 'false' otherwise.
         */
        bool operator==(const Camera &rhs) const;

    private:

        /** Updates all members using the camera parameters */
        void updateMembers(void);

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

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAOPENGL_H_INCLUDED */
