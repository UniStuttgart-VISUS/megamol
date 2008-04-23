/*
 * AbstractCameraController.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCAMERACONTROLLER_H_INCLUDED
#define VISLIB_ABSTRACTCAMERACONTROLLER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SmartPtr.h"
#include "vislib/CameraParameters.h"


namespace vislib {
namespace graphics {

    /**
     * Abstract base class for camera controller classes
     */
    class AbstractCameraController {
    public:

        /** 
         * Ctor. Initialises with a camera object. This controller object does
         * not take ownership of the memory of the camera object. The caller 
         * must therefore guarantee that the provided pointer remains valid 
         * until it is no longer used by this object.
         *
         * @param cameraParams The camera parameters object to control.
         */
        AbstractCameraController(
            const SmartPtr<CameraParameters>& cameraParams 
            = SmartPtr<CameraParameters>());

        /** Dtor. */
        virtual ~AbstractCameraController(void);

        /**
         * Answer the parameters object of the associated camera object. Must
         * not be called if 'IsCameraValid' returns 'false'.
         *
         * @return The parameters object of the associated camera object.
         */
        SmartPtr<CameraParameters>& CameraParams(void);

        /**
         * Answer the parameters object of the associated camera object. Must
         * not be called if 'IsCameraValid' returns 'false'.
         *
         * @return The parameters object of the associated camera object.
         */
        const SmartPtr<CameraParameters>& CameraParams(void) const;

        /**
         * Answer weather the associated 'Camera' object is valid.
         *
         * @return 'true' if the associated 'Camera' object is valid, 
         *         'false' otherwise.
         */
        inline bool IsCameraParamsValid(void) const {
            return !this->cameraParams.IsNull();
        }

        /**
         * Associates this controller with a new camera object. This 
         * controller object does not take ownership of the memory of the 
         * camera object. The caller must therefore guarantee that the 
         * provided pointer remains valid until it is no longer used by this 
         * object.
         *
         * @param cameraParams The camera parameters object to control.
         */
        void SetCameraParams(const SmartPtr<CameraParameters>& cameraParams);

    private:

        /** The associated 'Camera' object. */
        SmartPtr<CameraParameters> cameraParams;

    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCAMERACONTROLLER_H_INCLUDED */

