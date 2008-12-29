/*
 * CameraOpenGL.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAOPENGL_H_INCLUDED
#define VISLIB_CAMERAOPENGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
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
        void glMultProjectionMatrix(void) const;

        /**
         * Multiplies the current openGL matrix with the viewing matrix of this
         * camera.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void glMultViewMatrix(void) const;


        /**
         * Replace the current projection matrix and the current modelview 
         * matrix with the projection matrix respectively view matrixx of 
         * this camera. Note, that the current matrices will not be pushed onto
         * the stack, but replaced.
         *
         * The matrix mode after calling this method is GL_MODELVIEW.
         */
        inline void glSetMatrices(void) const {
            this->glSetProjectionMatrix();
            this->glSetViewMatrix();
        }

        /**
         * Replace the current projection matrix with the projection matrix of 
         * this camera. Note, that the current matrix will not be pushed onto
         * the stack, but replaced.
         *
         * The matrix mode after calling this method is GL_PROJECTION.
         */
        void glSetProjectionMatrix(void) const;

        /**
         * Replace the current modelview matrix with the view matrix of 
         * this camera. Note, that the current matrix will not be pushed onto
         * the stack, but replaced.
         *
         * The matrix mode after calling this method is GL_MODELVIEW.
         */
        void glSetViewMatrix(void) const;

        /**
         * Answer the projection matrix of the frustum of this camera.
         *
         * @param mat Points to an array of 16 floats receiving the matrix.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void ProjectionMatrix(float *mat) const;

        /**
         * Answer the view matrix of the frustum of this camera.
         *
         * @param mat Points to an array of 16 floats receiving the matrix.
         *
         * throws IllegalStateException if no beholder is associated with this
         *        Camera.
         */
        void ViewMatrix(float *mat) const;

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
        void updateMembers(void) const;

        /** viewing frustum minimal x */
        mutable SceneSpaceType left;

        /** viewing frustum maximal x */
        mutable SceneSpaceType right;

        /** viewing frustum minimal y */
        mutable SceneSpaceType bottom;

        /** viewing frustum maximal y */
        mutable SceneSpaceType top;

        /** viewing frustum minimal z */
        mutable SceneSpaceType nearClip;

        /** viewing frustum maximal z */
        mutable SceneSpaceType farClip;

        /** projection viewer position */
        mutable math::Point<SceneSpaceType, 3> pos;

        /** projection viewing direction */ 
        mutable math::Vector<SceneSpaceType, 3> lookDir;

        /** projection up vector */
        mutable math::Vector<SceneSpaceType, 3> up;

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAOPENGL_H_INCLUDED */
