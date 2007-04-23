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
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/graphicstypes.h"
#include "vislib/Beholder.h"
#include "vislib/mathtypes.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"
#include "vislib/Rectangle.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"
#include "vislib/Cuboid.h"
#include <float.h>


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene camera
     */
    class Camera {
    public:
        
        /** possible values for the projection type */
        enum ProjectionType {
            MONO_PERSPECTIVE = 0,
            MONO_ORTHOGRAPHIC,
            STEREO_PARALLEL,
            STEREO_OFF_AXIS,
            STEREO_TOE_IN
        };

        /** possible values for stereo eyes */
        enum StereoEye {
            LEFT_EYE = 0,
            RIGHT_EYE = 1
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
        Camera(Beholder *beholder);

        /**
         * dtor
         */
        virtual ~Camera(void);

        /**
         * Calculates the optimal clipping distances for the giving bounding
         * box, based on the position of the associated beholder.
         *
         * @param box The bounding box of the scene content.
         * @param minNear The minimal value for the near clipping plane
         * @param maxFar The maximal value for the far clipping plane
         */
        void CalcClipDistances(const vislib::math::Cuboid<SceneSpaceType> &box,
            SceneSpaceType minNear = 0.01f, SceneSpaceType maxFar = FLT_MAX);

        /** 
         * Returns aperture Angle of the camera along the y axis.
         *
         * @return The aperture Angle
         */
        inline math::AngleDeg GetApertureAngle(void) { 
            return math::AngleRad2Deg(this->halfApertureAngle * 2.0f);
        }

        /**
         * Returns the half aperture Angle of the camera along the y axis.
         *
         * @return The half aperture Angel in radians.
         */
        inline math::AngleRad GetHalfApertureAngleRad(void) {
            return this->halfApertureAngle;
        }

        /** 
         * Returns distance of the far clipping plane 
         *
         * @return distance of the far clipping plane
         */
        inline SceneSpaceType GetFarClipDistance(void) { 
            return this->farClip;
        }

        /** 
         * Returns focal distance for stereo images 
         *
         * @return The focal distance
         */
        inline SceneSpaceType GetFocalDistance(void) { 
            return this->focalDistance;
        }

        /** 
         * Returns distance of the near clipping plane 
         *
         * @return The distance of the near clipping plane
         */
        inline SceneSpaceType GetNearClipDistance(void) { 
            return this->nearClip;
        }

        /** 
         * Returns eye disparity value for stereo images 
         *
         * @return The eye disparity value
         */
        inline SceneSpaceType GetStereoDisparity(void) { 
            return this->halfStereoDisparity * 
                static_cast<SceneSpaceType>(2.0);
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
         * Returns the eye for stereo projection
         *
         * @return The eye for stereo projection
         */
        inline StereoEye GetStereoEye(void) {
            return this->eye;
        }

        /** 
         * Returns Width of the virtual camera image 
         *
         * @return The width of the virtual camera image 
         */
        inline ImageSpaceType GetVirtualWidth(void) { 
            return static_cast<ImageSpaceType>(this->virtualHalfWidth * 2.0);
        }

        /** 
         * Returns Height of the virtual camera image 
         *
         * @return The height of the virtual camera image
         */
        inline ImageSpaceType GetVirtualHeight(void) { 
            return static_cast<ImageSpaceType>(this->virtualHalfHeight * 2.0);
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
        void SetFarClipDistance(SceneSpaceType farClip);

        /** 
         * Sets the focal distance for stereo images and sets the 
         * memberChanged flag.
         * Values equal or less then Zero will be clamped to a small positive 
         * value.
         *
         * @param focalDistance The focal distance
         */
        void SetFocalDistance(SceneSpaceType focalDistance);

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
        void SetNearClipDistance(SceneSpaceType nearClip);

        /** 
         * Sets the eye disparity value for stereo images and sets the 
         * memberChanged flag.
         * If a negative value is supplied, it's absolute value is used.
         *
         * @param stereoDisparity The eye disparity value
         */
        void SetStereoDisparity(SceneSpaceType stereoDisparity);

        /**
         * Sets the type of stereo projection and sets the memberChanged flag.
         *
         * @param stereoProjectionType The type of stereo projection
         */
        void SetProjectionType(ProjectionType projectionType);

        /**
         * Sets the eye for stereo projection and sets the memberChanged flag.
         * This value has no effect if a mono projection is used.
         *
         * @param eye The new eye to be set.
         */
        void SetStereoEye(StereoEye eye);

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
        void SetVirtualWidth(ImageSpaceType virtualWidth);

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
        void SetVirtualHeight(ImageSpaceType virtualHeight);

        /**
         * Return the tile rectangle of the camera.
         *
         * @return The tile rectangle.
         */
        inline const math::Rectangle<ImageSpaceType> & GetTileRectangle(void) const {
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
            this->tileRect.Set(0, 0,
                this->virtualHalfWidth * 2.0f, this->virtualHalfHeight * 2.0f);
        }

        /**
         * Sets the tile rectangle of the camera and sets the memberChanged 
         * flag.
         *
         * @param tileRect The new tile rectangle for the camera.
         */
        template <class Tp, class Sp > 
        void SetTileRectangle(const math::AbstractRectangle<Tp, Sp> &tileRect);

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
        void SetBeholder(Beholder *beholder);

        /**
         * Answer the associated beholder of this camera.
         *
         * @return Pointer to the associated beholder of this camera.
         */
        inline Beholder * GetBeholder(void) {
            return this->beholder;
        }

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
        void CalcFrustumParameters(SceneSpaceType &outLeft,
            SceneSpaceType &outRight, SceneSpaceType &outBottom,
            SceneSpaceType &outTop, SceneSpaceType &outNearClip,
            SceneSpaceType &outFarClip);

        /**
         * Calculates and returns all parameters necessary to set up the view
         * matrix of this camera.
         *
         * @param outPosition Returns the position of the camera.
         * @param outFront Returns the vector in viewing direction of the camera.
         *                 The returned vector may be not normalized.
         * @param outUp Returns the up vector of the camera.
         *
         * @throws IllegalStateException if this camera is not associated with a
         *         Beholder.
         */
        void CalcViewParameters(
            math::Point<SceneSpaceType, 3> &outPosition,
            math::Vector<SceneSpaceType, 3> &outFront,
            math::Vector<SceneSpaceType, 3> &outUp);

        /**
         * Answer wether the view or frustum parameters need to be recalculated.
         *
         * @return true if the parameters need to be recalculated, false 
         *         otherwise.
         */
        inline bool NeedUpdate(void) {
            return this->membersChanged || (this->updateCounter == 0)
                || (this->beholder == NULL) 
                || (this->updateCounter != this->beholder->GetUpdateCounterValue());
        }

        /**
         * Clears all update flaggs.
         */
        inline void ClearUpdateFlaggs(void) {
            this->membersChanged = false;
            if (this->beholder) {
                this->updateCounter = this->beholder->GetUpdateCounterValue();
            }
        }

    private:

        /**
         * Sets default values for all members, except updateCounter, and
         * holder which are not changed.
         */
        void SetDefaultValues(void);

        /** half aperture Angle in radians of the camera along the y axis */
        math::AngleRad halfApertureAngle;

        /** Pointer to the holder of the currently attached beholder */
        Beholder *beholder;

        /** distance of the far clipping plane */
        SceneSpaceType farClip;

        /** focal distance for stereo images */
        SceneSpaceType focalDistance;

        /** distance of the near clipping plane */
        SceneSpaceType nearClip;

        /** half eye disparity value for stereo images */
        SceneSpaceType halfStereoDisparity;

        /** type of stereo projection */
        ProjectionType projectionType;

        /** eye for stereo projections */
        StereoEye eye;

        /** Half width of the virtual camera image along the right vector*/
        ImageSpaceType virtualHalfWidth;

        /** Half height of the virtual camera image along the up vector */
        ImageSpaceType virtualHalfHeight;

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
        math::Rectangle<ImageSpaceType> tileRect;
    };

    
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

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERA_H_INCLUDED */
