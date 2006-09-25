/*
 * Camera.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERA_H_INCLUDED
#define VISLIB_CAMERA_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Beholder.h"
#include "vislib/mathtypes.h"
#include "vislib/IllegalStateException.h"


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene camera
     */
    class Camera {
    public:

        /** This type is used for values in scene space */
        typedef float SceneSpaceValue;

        /** This type is used for values in image space */
        typedef unsigned int ImageSpaceValue;
        
        /** possible values for the stereo projection type */
        enum StereoProjectionType {
            PARALLEL_PROJECTION = 0,
            OFF_AXIS_PROJECTION,
            TOE_IN_PROJECTION
        };

        /**
         * default ctor
         */
        Camera(void);

        /**
         * default dtor
         */
        ~Camera(void);

        /*
         * TODO: Implement
         */

        /**
         * Associates this camera with the beholder specified.
         * Ownership of the beholder is not changed, thus the caller must 
         * ensure that the beholder object is valid as long as it is
         * associated with this camera.
         *
         * @param beholder Pointer to the new beholder to be associated with 
         *                 this camera
         *
         * @throws std::bad_alloc 
         */
        template <class Tp > void SetBeholder(Beholder<Tp> *beholder);

        /**
         * Answer the associated beholder of this camera.
         *
         * @return Pointer to the associated beholder of this camera.
         *
         * @throws IllegalStateException if the Getter has been instanciated
         */
        template <class Tp > Beholder<Tp> * GetBeholder(void);

    private:

        /**
         * Abstract base class as beholder facade
         */
        class AbstractBeholderHolder {
        public:
            /**
             * dtor
             */
            virtual ~AbstractBeholderHolder(void) {
            }
        };

        /**
         * template implementation of the beholder facade
         */
        template <class T > class BeholderHolder: public AbstractBeholderHolder {
        public:
            /**
             * ctor
             *
             * @param beholder The beholder to hold
             */
            BeholderHolder(Beholder<T> *beholder) {
                this->beholder = beholder;
            }

            /**
             * answer the beholder hold.
             *
             * @return The beholder hold.
             */
            operator Beholder<T> *() {
                return this->beholder;
            }

        private:
            /** the beholder hold */
            Beholder<T> *beholder;
        };

        /** 
         * aperture Angle of the camera
         * TODO: Think of the meaning of this value
         */
        math::AngleDeg apertureAngle;

        /** 
         * Pointer to the currently attached beholder 
         */
        AbstractBeholderHolder *holder;

        /** distance of the far clipping plane */
        SceneSpaceValue farClip;

        /** focal distance for stereo images */
        SceneSpaceValue focalDistance;

        /** distance of the near clipping plane */
        SceneSpaceValue nearClip;

        /** eye disparity value for stereo images */
        SceneSpaceValue stereoDisparity;

        /** type of stereo projection */
        StereoProjectionType stereoProjectionType;

        /** Width of the virtual camera image */
        ImageSpaceValue virtualWidth;

        /** Height of the virtual camera image */
        ImageSpaceValue virtualHeight;
    };


    /*
     * Camera::SetBeholder
     */
    template <class Tp > void Camera::SetBeholder(Beholder<Tp> *beholder) {
        delete this->holder;
    }

    /*
     * Camera::GetBeholder
     */
    template <class Tp > Beholder<Tp> * Camera::GetBeholder(void) {
        Camera::BeholderHolder<Tp> *holder 
            = dynamic_cast<Camera::BeholderHolder<Tp> *>(this->holder);
        if (!holder) {
            throw IllegalStateException(
                "Camera::GetBeholder instantiated with incompatible type", 
                __FILE__, __LINE__);
        }
        return *holder;
    }

} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_CAMERA_H_INCLUDED */
