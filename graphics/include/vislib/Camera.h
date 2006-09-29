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
#include "vislib/AbstractRectangle.h"
#include "vislib/Rectangle.h"
#include "vislib/Point3D.h"
#include "vislib/Vector3D.h"


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
        
        /** possible values for the projection type */
        enum ProjectionType {
            MONO_PERSPECTIVE = 0,
            MONO_ORTHOGRAPHIC,
            STEREO_PARALLEL_LEFT,
            STEREO_PARALLEL_RIGHT,
            STEREO_OFF_AXIS_LEFT,
            STEREO_OFF_AXIS_RIGHT,
            STEREO_TOE_IN_LEFT,
            STEREO_TOE_IN_RIGHT
        };

        /**
         * default ctor
         */
        Camera(void);

        /**
         * copy ctor
         *
         * @param rhs Camera values will be copied from.
         */
        Camera(const Camera &rhs);

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
        virtual ~Camera(void);

        /** 
         * Returns aperture Angle of the camera along the y axis.
         *
         * @return The aperture Angle
         */
        inline math::AngleDeg GetApertureAngle(void) { 
            return math::AngleRad2Deg(this->halfApertureAngle * 2.0f);
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
            return this->halfStereoDisparity * 2.0f;
        }

        /** 
         * Returns type of the projection 
         *
         * @return The type of the projection
         */
        inline ProjectionType GetProjectionType(void) { 
            return this->projectionType;
        }

        /** 
         * Returns Width of the virtual camera image 
         *
         * @return The width of the virtual camera image 
         */
        inline ImageSpaceValue GetVirtualWidth(void) { 
            return this->virtualHalfWidth * 2.0f;
        }

        /** 
         * Returns Height of the virtual camera image 
         *
         * @return The height of the virtual camera image
         */
        inline ImageSpaceValue GetVirtualHeight(void) { 
            return this->virtualHalfHeight * 2.0f;
        }

        /**
         * Sets the aperture angle and sets the memberChanged flag.
         *
         * @param apertureAngle the aperture angle.
         *
         * @throws IllegalParamException if the angle specified is not more 
         *         then Zero, and less then 180.0 degree.
         */
        void SetApertureAngle(math::AngleDeg apertureAngle);

        /**
         * Sets the distance of the far clipping plane and sets the 
         * memberChanged flag.
         * Values equal or less to the distance of the current near clipping 
         * plane will be clamped to the value of the current near clipping 
         * plane + a positive delta.
         *
         * @param farClip the distance of the far clipping plane
         */
        void SetFarClipDistance(SceneSpaceValue farClip);

        /** 
         * Sets the focal distance for stereo images and sets the 
         * memberChanged flag.
         * Values equal or less then Zero will be clamped to a small positive 
         * value.
         *
         * @param focalDistance The focal distance
         */
        void SetFocalDistance(SceneSpaceValue focalDistance);

        /** 
         * Sets distance of the near clipping plane and sets the memberChanged 
         * flag.
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
         * Sets the eye disparity value for stereo images and sets the 
         * memberChanged flag.
         * If a negative value is supplied, it's absolute value is used.
         *
         * @param stereoDisparity The eye disparity value
         */
        void SetStereoDisparity(SceneSpaceValue stereoDisparity);

        /**
         * Sets the type of stereo projection and sets the memberChanged flag.
         *
         * @param stereoProjectionType The type of stereo projection
         */
        void SetProjectionType(ProjectionType projectionType);

        /** 
         * Sets the width of the virtual camera image and sets the 
         * memberChanged flag.
         * Values equal or less then Zero will be clamped to 1.
         * If the left value of the tile rectangle is Zero and the right value
         * is equal to the current virtual image width, the right value of the
         * tile rectangle is also set to the virtual width parameter.
         *
         * @param virtualWidth The width of the virtual camera image
         */
        void SetVirtualWidth(ImageSpaceValue virtualWidth);

        /** 
         * Sets the height of the virtual camera image and sets the 
         * memberChanged flag.
         * Values equal or less then Zero will be clamped to 1.
         * If the bottom value of the tile rectangle is Zero and the top value
         * is equal to the current virtual image height, the top value of the
         * tile rectangle is also set to the virtual height parameter.
         *
         * @param virtualHeight The height of the virtual camera image 
         */
        void SetVirtualHeight(ImageSpaceValue virtualHeight);

        /**
         * Return the tile rectangle of the camera.
         *
         * @return The tile rectangle.
         */
        inline const math::Rectangle<ImageSpaceValue> & GetTileRectangle(void) const {
            return this->tileRect;
        }

        /**
         * Resets the tile rectangle of the camera to the size of the whole 
         * virtual camera image.
         * Left and bottom will be set to Zero, right will be set to the width
         * of the virtual camera image, and top will be set to the height of 
         * the virtual camera image.
         */
        inline void ResetTileRectangle(void) {
            this->tileRect.Set(static_cast<ImageSpaceValue>(0), 
                static_cast<ImageSpaceValue>(0), 
                this->virtualHalfWidth * 2.0f, this->virtualHalfHeight * 2.0f);
        }

        /**
         * Sets the tile rectangle of the camera and sets the memberChanged 
         * flag.
         *
         * @param tileRect The new tile rectangle for the camera.
         */
        template <class Tp, class Sp > void SetTileRectangle(const math::AbstractRectangle<Tp, Sp> &tileRect);

        /**
         * Associates this camera with the beholder specified and resets the
         * update counter value of this camera.
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
         * Assignment operator
         *
         * @param rhs Camera values will be copied from
         *
         * @return Reference to this.
         */
        Camera & operator=(const Camera &rhs);

    protected:

        /**
         * Calculates and returns all parameters necessary to set up a 
         * projection matrix modelling the viewing frustum of this camera.
         *
         * @param outLeft Receives the minimal x value of the frustum on the 
         *                near clipping plane.
         * @param outRight Receives the maximal x value of the frustum on the 
         *                 near clipping plane.
         * @param outBottom Receives the minimal y value of the frustum on the 
         *                  near clipping plane.
         * @param outTop Receives the maximal y value of the frustum on the 
         *               near clipping plane.
         * @param outNearClip Receives the distance of the near clipping plane 
         *                    from the camera position.
         * @param outFraClip Receives the distance of the far clipping plane
         *                   from the camera position.
         *
         * @throws IllegalStateException if this camera is not associated with a
         *         Beholder.
         */
        void CalcFrustumParameters(SceneSpaceValue &outLeft,
            SceneSpaceValue &outRight, SceneSpaceValue &outBottom,
            SceneSpaceValue &outTop, SceneSpaceValue &outNearClip,
            SceneSpaceValue &outFarClip);

        /**
         * Calculates and returns all parameters necessary to set up the view
         * matrix of this camera.
         *
         * @param outPosition Returns the position of the camera.
         * @param outFront Returns the vector in viewing direction of the camera.
         * @param outUp Returns the up vector of the camera.
         *
         * @throws IllegalStateException if this camera is not associated with a
         *         Beholder.
         */
        void CalcViewParameters(
            math::Point3D<SceneSpaceValue> &outPosition,
            math::Vector3D<SceneSpaceValue> &outFront,
            math::Vector3D<SceneSpaceValue> &outUp);

        /**
         * Answer wether the view or frustum parameters need to be recalculated.
         *
         * @return true if the parameters need to be recalculated, false 
         *         otherwise.
         */
        inline bool NeedUpdate(void) {
            return this->membersChanged || (this->updateCounter == 0)
                || (this->holder == NULL) 
                || (this->updateCounter != this->holder->GetUpdateCounterValue());
        }

        /**
         * Clears all update flaggs.
         */
        inline void ClearUpdateFlaggs(void) {
            this->membersChanged = false;
            if (this->holder) {
                this->updateCounter = this->holder->GetUpdateCounterValue();
            }
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
             * Clones this.
             *
             * @return Pointer to a new object with identically values as this.
             */
            virtual AbstractBeholderHolder * Clone(void) = 0;

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

            /** behaves like AbstractBeholderHolder::Clone */
            virtual AbstractBeholderHolder * Clone(void) {
                return new BeholderHolder<T>(this->beholder);
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

        /** half aperture Angle in radians of the camera along the y axis */
        math::AngleDeg halfApertureAngle;

        /** Pointer to the holder of the currently attached beholder */
        AbstractBeholderHolder *holder;

        /** distance of the far clipping plane */
        SceneSpaceValue farClip;

        /** focal distance for stereo images */
        SceneSpaceValue focalDistance;

        /** distance of the near clipping plane */
        SceneSpaceValue nearClip;

        /** half eye disparity value for stereo images */
        SceneSpaceValue halfStereoDisparity;

        /** type of stereo projection */
        ProjectionType projectionType;

        /** Half width of the virtual camera image along the right vector*/
        ImageSpaceValue virtualHalfWidth;

        /** Half height of the virtual camera image along the up vector */
        ImageSpaceValue virtualHalfHeight;

        /** 
         * The camera update counter value to be compared with the update 
         * counter value of the beholder.
         */
        unsigned int updateCounter;

        /** 
         * Flag to indicate that the values of at least one member might have
         * changed.
         */
        bool membersChanged;

        /** The selected clip tile rectangle of the virtual camera image */
        math::Rectangle<ImageSpaceValue> tileRect;
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
        this->updateCounter = 0;
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

    
    /*
     * Camera::SetTileRectangle
     */
    template <class Tp, class Sp > 
    void Camera::SetTileRectangle(const math::AbstractRectangle<Tp, Sp> &tileRect) {
        this->tileRect = tileRect;
        this->membersChanged = true;
    }

} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_CAMERA_H_INCLUDED */
