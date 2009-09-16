/*
 * D3DCamera.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DCAMERA_H_INCLUDED
#define VISLIB_D3DCAMERA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Camera.h"
#include "vislib/D3DMatrix.h"
#include "vislib/D3DPoint3D.h"
#include "vislib/D3DVector3D.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * This is a camera specialisation for use with Direct3D.
     */
    class D3DCamera : public Camera {

    public:

        /** Ctor. */
        D3DCamera(void);

        /** 
         * Ctor. Initialises the camera with the given camera parameters.
         *
         * @param params The camera parameters to be used. Must not be NULL.
         */
        D3DCamera(const SmartPtr<CameraParameters>& params);

        /** Dtor. */
        virtual ~D3DCamera(void);

        /**
         * Set the projection matrix for the camera as computed by 
         * CalcProjectionMatrix() as current projection transform of the 
         * specified device.
         *
         * @param device The Direct3D 9 device to set the projection transform of.
         *
         * @return The return value of the underlying SetTransform() call on 
         *         'device'.
         */
        HRESULT ApplyProjectionTransform(IDirect3DDevice9 *device) const;

        /**
         * Set the view matrix for the camera as computed by 
         * CalcViewMatrix() as current view transform of the specified
         * device.
         *
         * @param device The Direct3D 9 device to set the view transform of.
         *
         * @return The return value of the underlying SetTransform() call on 
         *         'device'.
         */
        HRESULT ApplyViewTransform(IDirect3DDevice9 *device) const;

        /**
         * Answer the projection matrix for the camera.
         *
         * @param outMatrix    This parameter receives the matrix.
         * @param isLeftHanded If true, return the matrix for a left-handed 
         *                     coordinate system (which is D3D default), 
         *                     otherwise construct a matrix for a right-handed
         *                     system.
         *
         * @return The matrix 'outMatrix' is returned for convenience.
         */
        D3DMatrix& CalcProjectionMatrix(D3DMatrix& outMatrix, 
            const bool isLeftHanded = true) const;

        /**
         * Answer the view matrix for the camera.
         *
         * @param outMatrix    This parameter receives the matrix.
         * @param isLeftHanded If true, return the matrix for a left-handed 
         *                     coordinate system (which is D3D default), 
         *                     otherwise construct a matrix for a right-handed
         *                     system.
         *
         * @return The matrix 'outMatrix' is returned for convenience.
         */
        D3DMatrix& CalcViewMatrix(D3DMatrix& outMatrix,
            const bool isLeftHanded = true) const;

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        D3DCamera& operator =(const D3DCamera& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const D3DCamera& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const D3DCamera& rhs) const {
            return !(*this == rhs);
        }

    protected:

        /**
         * Update the cached view parameters if necessary.
         *
         * @return true if the cached parameters have been updated, 
         *         false otherwise.
         */
        bool updateCache(void) const;
    
        /** Cached view frustum. */
        mutable SceneSpaceViewFrustum cacheFrustum;

        /** Cached look at point (for the current eye). */ 
        mutable D3DPoint3D cacheAt;

        /** Cached viewer position (for the current eye). */
        mutable D3DPoint3D cacheEye;

        /** Cached up vector (for the current eye). */
        mutable D3DVector3D cacheUp;

    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DCAMERA_H_INCLUDED */
