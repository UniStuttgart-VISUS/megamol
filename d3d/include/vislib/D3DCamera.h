/*
 * D3DCamera.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DCAMERA_H_INCLUDED
#define VISLIB_D3DCAMERA_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Camera.h"


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
    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DCAMERA_H_INCLUDED */

