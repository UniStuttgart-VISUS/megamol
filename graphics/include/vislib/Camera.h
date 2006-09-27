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
#include "vislib/memutils.h"


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene camera
     */
    class Camera {
    public:

        /** This type is used for values in scene space */
        typedef float SceneSpaceValue;

        /** 
         * This type is used for values in image space 
         * Implementation note: using float instead of unsigned int to be able 
         * to place elements with subpixel precision
         */
        typedef float ImageSpaceValue;
        
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
         * ctor
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
        template <class Tp > Camera(Beholder<Tp> *beholder);

        /**
         * dtor
         */
        ~Camera(void);

        /** 
         * Returns aperture Angle of the camera along the y axis.
         *
         * @return The aperture Angle
         */
        inline math::AngleDeg GetApertureAngle(void) { 
            return this->apertureAngle;
        }

        /** 
         * Returns distance of the far clipping plane 
         *
         * @return distance of the far clipping plane
         */
        inline SceneSpaceValue GetFarClipDistance(void) { 
            return this->farClip;
        }

        /** 
         * Returns focal distance for stereo images 
         *
         * @return The focal distance
         */
        inline SceneSpaceValue GetFocalDistance(void) { 
            return this->focalDistance;
        }

        /** 
         * Returns distance of the near clipping plane 
         *
         * @return The distance of the near clipping plane
         */
        inline SceneSpaceValue GetNearClipDistance(void) { 
            return this->nearClip;
        }

        /** 
         * Returns eye disparity value for stereo images 
         *
         * @return The eye disparity value
         */
        inline SceneSpaceValue GetStereoDisparity(void) { 
            return this->stereoDisparity;
        }

        /** 
         * Returns type of stereo projection 
         *
         * @return The type of stereo projection
         */
        inline StereoProjectionType GetStereoProjectionType(void) { 
            return this->stereoProjectionType;
        }

        /** 
         * Returns Width of the virtual camera image 
         *
         * @return The width of the virtual camera image 
         */
        inline ImageSpaceValue GetVirtualWidth(void) { 
            return this->virtualWidth;
        }

        /** 
         * Returns Height of the virtual camera image 
         *
         * @return The height of the virtual camera image
         */
        inline ImageSpaceValue GetVirtualHeight(void) { 
            return this->virtualHeight;
        }

        /**
         * Sets the aperture angle and increments the update counter.
         *
         * @param apertureAngle the aperture angle.
         *
         * @throws IllegalParamException if the angle specified is not more 
         *         then Zero, and less then 180.0 degree.
         */
        void SetApertureAngle(math::AngleDeg apertureAngle);

        /**
         * Sets the distance of the far clipping plane and increments the 
         * update counter.
         * Values equal or less to the distance of the current near clipping 
         * plane will be clamped to the value of the current near clipping 
         * plane + a positive delta.
         *
         * @param farClip the distance of the far clipping plane
         */
        void SetFarClipDistance(SceneSpaceValue farClip);

        /** 
         * Sets the focal distance for stereo images and increments the update 
         * counter.
         * Values equal or less then Zero will be clamped to a small positive 
         * value.
         *
         * @param focalDistance The focal distance
         */
        void SetFocalDistance(SceneSpaceValue focalDistance);

        /** 
         * Sets distance of the near clipping plane and increments the update 
         * counter.
         * Values equal or less then Zero will be clamped to a small positive 
         * value.
         * If the new value is equal or larger then the distance of the corrent
         * far clipping plane, the far clipping plane is moved to a distance
         * slightly larger then the given value for the near clipping plane.
         *
         * @param nearClip The distance of the near clipping plane 
         */
        void SetNearClipDistance(SceneSpaceValue nearClip);

        /** 
         * Sets the eye disparity value for stereo images and increments the 
         * update counter.
         * If a negative value is supplied, it's absolute value is used.
         *
         * @param stereoDisparity The eye disparity value
         */
        void SetStereoDisparity(SceneSpaceValue stereoDisparity);

        /**
         * Sets the type of stereo projection and increments the update counter.
         *
         * @param stereoProjectionType The type of stereo projection
         */
        void SetStereoProjectionType(StereoProjectionType stereoProjectionType);

        /** 
         * Sets the width of the virtual camera image and increments the 
         * update counter.
         * Values equal or less then Zero will be clamped to 1.
         *
         * @param virtualWidth The width of the virtual camera image
         */
        void SetVirtualWidth(ImageSpaceValue virtualWidth);

        /** 
         * Sets the height of the virtual camera image and increments the 
         * update counter.
         * Values equal or less then Zero will be clamped to 1.
         *
         * @param virtualHeight The height of the virtual camera image 
         */
        void SetVirtualHeight(ImageSpaceValue virtualHeight);

        /**
         * returns the value of the updateCounter.
         *
         * @return The value of the updateCounter
         */
        inline const unsigned int GetUpdateCounterValue(void) const {
            return this->updateCounter;
        }

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
         *         with another type as the beholder has been.
         */
        template <class Tp > Beholder<Tp> * GetBeholder(void);

        /**
         * returns the position of the associated beholder in world coordinates.
         *
         * @return The position
         *
         * @throws 
         */
        inline void ReturnBeholderPosition(math::Point3D<SceneSpaceValue> &outPosition) const {
            this->holder->ReturnPosition(outPosition);
        }

        /**
         * returns the look at point of the associated beholder in world coordinates.
         *
         * @return The look at point
         */
        inline void ReturnBeholderLookAt(math::Point3D<SceneSpaceValue> &outLookAt) const {
            this->holder->ReturnLookAt(outLookAt);
        }

        /**
         * returns the front vector of the associated beholder.
         *
         * @return The front vector
         */
        inline void ReturnBeholderFrontVector(math::Vector3D<SceneSpaceValue> &outFront) const {
            this->holder->ReturnFrontVector(outFront);
        }

        /**
         * returns the right vector of the associated beholder.
         *
         * @return The right vector
         */
        inline void ReturnBeholderRightVector(math::Vector3D<SceneSpaceValue> &outRight) const {
            this->holder->ReturnRightVector(outRight);
        }

        /**
         * returns the up vector of the associated beholder.
         *
         * @return The up vector
         */
        inline void ReturnBeholderUpVector(math::Vector3D<SceneSpaceValue> &outUp) const {
            this->holder->ReturnUpVector(outUp);
        }

        /**
         * returns the value of the updateCounter of the beholder
         *
         * @return The value of the updateCounter of the beholder
         */
        inline const unsigned int GetBeholderUpdateCounterValue(void) const {
            return this->holder->GetUpdateCounterValue();
        }

    private:

        /**
         * Sets default values for all members, except updateCounter, and
         * holder which are not changed.
         */
        void SetDefaultValues(void);

        /**
         * Abstract base class as beholder facade following the crowbar pattern
         */
        class AbstractBeholderHolder {
        public:
            /**
             * Answer the update counter value of the beholder hold.
             *
             * @return The update counter value of the beholder hold.
             */
            virtual unsigned int GetUpdateCounterValue(void) = 0;
        
            /**
             * returns the position of the beholder hold in world coordinates.
             *
             * @param outPosition The position
             */
            virtual void ReturnPosition(math::Point3D<SceneSpaceValue> &outPosition) const = 0;

            /**
             * returns the look at point of the beholder hold in world coordinates.
             *
             * @param outLootAt The look at point
             */
            virtual void ReturnLookAt(math::Point3D<SceneSpaceValue> &outLookAt) const = 0;

            /**
             * returns the front vector of the beholder hold.
             *
             * @param outFront The front vector
             */
            virtual void ReturnFrontVector(math::Vector3D<SceneSpaceValue> &outFront) const = 0;

            /**
             * returns the right vector of the beholder hold.
             *
             * @param outRight The right vector
             */
            virtual void ReturnRightVector(math::Vector3D<SceneSpaceValue> &outRight) const = 0;

            /**
             * returns the up vector of the beholder hold.
             *
             * @param outUp The up vector
             */
            virtual void ReturnUpVector(math::Vector3D<SceneSpaceValue> &outUp) const = 0;

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
            operator Beholder<T> *(void) {
                return this->beholder;
            }

            /** behaves like AbstractBeholderHolder::GetUpdateCounterValue */
            virtual unsigned int GetUpdateCounterValue(void) {
                return this->beholder->GetUpdateCounterValue();
            }

            /** behaves like AbstractBeholderHolder::GetPosition */
            virtual void ReturnPosition(math::Point3D<SceneSpaceValue> &outPosition) const {
                outPosition = this->beholder->GetPosition();
            }

            /** behaves like AbstractBeholderHolder::GetLookAt */
            virtual void ReturnLookAt(math::Point3D<SceneSpaceValue> &outLookAt) const {
                outLookAt = this->beholder->GetLookAt();
            }

            /** behaves like AbstractBeholderHolder::GetFrontVector */
            virtual void ReturnFrontVector(math::Vector3D<SceneSpaceValue> &outFront) const {
                outFront = this->beholder->GetFrontVector();
            }

            /** behaves like AbstractBeholderHolder::GetRightVector */
            virtual void ReturnRightVector(math::Vector3D<SceneSpaceValue> &outRight) const {
                outRight = this->beholder->GetRightVector();
            }

            /** behaves like AbstractBeholderHolder::GetUpVector */
            virtual void ReturnUpVector(math::Vector3D<SceneSpaceValue> &outUp) const {
                outUp = this->beholder->GetUpVector();
            }

        private:
            /** 
             * the beholder hold 
             *
             * Implementation note: We do not own the beholder object, so do 
             * not destroy the memory in the destructor!
             */
            Beholder<T> *beholder; 
        };

        /** aperture Angle of the camera along the y axis */
        math::AngleDeg apertureAngle;

        /** Pointer to the holder of the currently attached beholder */
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

        /** The camera update counter value */
        unsigned int updateCounter;
    };


    /*
     * Camera::Camera
     */
     template <class Tp > Camera::Camera(Beholder<Tp> *beholder) : updateCounter(0) {
        this->holder = new Camera::BeholderHolder<Tp>(beholder);
        this->SetDefaultValues();
     }


    /*
     * Camera::SetBeholder
     */
    template <class Tp > void Camera::SetBeholder(Beholder<Tp> *beholder) {
        SAFE_DELETE(this->holder);
        this->holder = new Camera::BeholderHolder<Tp>(beholder);
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
