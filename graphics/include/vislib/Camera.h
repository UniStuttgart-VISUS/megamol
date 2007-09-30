/*
 * Camera.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERA_H_INCLUDED
#define VISLIB_CAMERA_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameters.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace graphics {


    /**
     * Base class for all camera implementations
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
         * Answers the parameters object.
         *
         * @return The parameters object.
         */
        SmartPtr<CameraParameters>& Parameters(void);

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
