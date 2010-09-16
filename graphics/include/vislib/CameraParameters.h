/*
 * CameraParameters.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAPARAMETERS_H_INCLUDED
#define VISLIB_CAMERAPARAMETERS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameterLimits.h"
#include "vislib/Cuboid.h"
#include "vislib/Dimension.h"
#include "vislib/graphicstypes.h"
#include "vislib/mathtypes.h"
#include "vislib/Point.h"
#include "vislib/Rectangle.h"
#include "vislib/Serialiser.h"
#include "vislib/Serialisable.h"
#include "vislib/SmartPtr.h"
#include "vislib/Vector.h"


namespace vislib {
namespace graphics {


    /**
     * Abstract base class of camera parameters used by the camera objects
     */
    class CameraParameters : public vislib::Serialisable {
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

        /** Ctor. */
        CameraParameters(void);

        /**
         * Copy ctor. 
         *
         * @param rhs The right hand side operand.
         */
        CameraParameters(const CameraParameters& rhs);

        /** Dtor. */
        virtual ~CameraParameters(void);

        /** 
         * Answer the aperture Angle in degree of the camera along the y axis
         * on the virtual view.
         *
         * @return The aperture Angle in degree of the camera along the y axis
         *         on the virtual view. 
         */
        inline math::AngleDeg ApertureAngle(void) const {
            return math::AngleRad2Deg(this->HalfApertureAngle()
                * static_cast<vislib::math::AngleRad>(2));
        }

        /**
         * Applies the limits to the stored values. This should be called
         * whenever the values stored in the limits object are changed. All
         * stored members may be changed.
         *
         * Note: The limits are also applied whenever a 'Set'-method is called.
         * If a 'Set'-method is called with parameter values violating the set
         * limits, the parameter values may be clipped to valid values or the
         * new values will not be set.
         */
        virtual void ApplyLimits(void) = 0;

        /**
         * Answers the auto focus offset
         *
         * @return The auto focus offset
         */
        virtual SceneSpaceType AutoFocusOffset(void) const = 0;

        /**
         * Calculates the clipping distances based on a bounding box cuboid
         * specified in world coordinates. (This methods implementation uses
         * 'SetClip', 'Position', and 'Front'.)
         *
         * @param bbox The bounding box in world coordinates.
         * @param border Additional distance of the clipping distances to the
         *               bounding box.
         */
        void CalcClipping(const math::Cuboid<SceneSpaceType>& bbox, 
            SceneSpaceType border);

        /**
         * Answer the coordinate system type of the camera.
         *
         * @return the coordinate system type of the camera.
         */
        virtual math::CoordSystemType CoordSystemType(void) const = 0;

        /**
         * Copies all values from 'src' into this object.
         *
         * @param src The source object to copy from.
         */
        inline void CopyFrom(const SmartPtr<CameraParameters> src) {
            this->CopyFrom(src.operator->());
        }

        /**
         * Copies all values from 'src' into this object.
         *
         * @param src The source object to copy from.
         */
        inline void CopyFrom(const CameraParameters *src) {
            this->SetApertureAngle(src->ApertureAngle());
            this->SetAutoFocusOffset(src->AutoFocusOffset());
            this->SetClip(src->NearClip(), src->FarClip());
            this->SetCoordSystemType(src->CoordSystemType());
            this->SetProjection(src->Projection());
            this->SetStereoParameters(src->StereoDisparity(), src->Eye(), src->FocalDistance());
            this->SetView(src->Position(), src->LookAt(), src->Up());
            this->SetVirtualViewSize(src->VirtualViewSize());
            this->SetTileRect(src->TileRect()); // set after virtual view size
            this->SetLimits(src->Limits()); // set as last
        }

        /**
         * Deserialise the object from 'serialiser'. The caller must ensure 
         * that the Serialiser is in an acceptable state to deserialise from.
         *
         * The camera parameter limits are NOT deserialised!
         *
         * @param serialiser The Serialiser to deserialise the object from.
         *
         * @throws Exception Implementing classes may throw exceptions to 
         *                   indicate an error or pass through exceptions thrown
         *                   by the Serialiser.
         */
        virtual void Deserialise(Serialiser& serialiser);

        /**
         * Answer the eye for stereo projections.
         *
         * @return The eye for stereo projections.
         */
        virtual StereoEye Eye(void) const = 0;

        /**
         * Calculates and returns the real eye looking direction taking stereo
         * projection mode and stereo disparity into account.
         *
         * @return The real eye looking direction.
         */
        virtual math::Vector<SceneSpaceType, 3> EyeDirection(void) const = 0;

        /**
         * Calculates and returns the real eye looking direction taking stereo
         * projection mode and stereo disparity into account.
         *
         * @return The real eye looking direction.
         */
        virtual math::Vector<SceneSpaceType, 3> EyeUpVector(void) const = 0;

        /**
         * Calculates and returns the real eye looking direction taking stereo
         * projection mode and stereo disparity into account. This vector
         * depends on the coordinate system type of the camera.
         *
         * @return The real eye looking direction.
         */
        virtual math::Vector<SceneSpaceType, 3> EyeRightVector(void) const = 0;

        /**
         * Calculates and returns the real eye position taking stereo 
         * projection mode and stereo disparity into account. This vector
         * depends on the coordinate system type of the camera.
         *
         * @return The real eye position.
         */
        virtual math::Point<SceneSpaceType, 3> EyePosition(void) const = 0;

        /** 
         * Answer the distance of the far clipping plane 
         *
         * @return The distance of the far clipping plane 
         */
        virtual SceneSpaceType FarClip(void) const = 0;

        /** 
         * Answer the focal distance for stereo images 
         *
         * @param autofocus If 'true' and the focus distance is in auto-focus
         *                  mode (0.0f) it returns the distance to the look-at
         *                  point instead of 0.0f.
         *
         * @return The focal distance for stereo images 
         */
        virtual SceneSpaceType FocalDistance(bool autofocus = true) const = 0;

        /** 
         * Answer the normalised front vector of the camera. 
         *
         * @return The normalised front vector of the camera. 
         */
        virtual const math::Vector<SceneSpaceType, 3>& Front(void) const = 0;

        /**
         * Answer the half aperture angle in radians.
         *
         * @return The half aperture angle in radians.
         */
        virtual math::AngleRad HalfApertureAngle(void) const = 0;

        /**
         * Answer the half stereo disparity.
         *
         * @return The half stereo disparity.
         */
        virtual SceneSpaceType HalfStereoDisparity(void) const = 0;

        /**
         * Answer whether this camera parameters object is similar with the
         * specified one. Similarity is given if the objects are equal or are
         * based on equal objects.
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if both objects are similar, 'false' otherwise.
         */
        virtual bool IsSimilar(const SmartPtr<CameraParameters> rhs) const = 0;

        /**
         * Answer the limits object used by this object.
         *
         * @return The limits object.
         */
        virtual SmartPtr<CameraParameterLimits> Limits(void) const = 0;

        /** 
         * Asnwer the look-at-point of the camera in world coordinates 
         *
         * @return The look-at-point of the camera in world coordinates 
         */
        virtual const math::Point<SceneSpaceType, 3>& LookAt(void) const = 0;

        /** 
         * Answer the distance of the near clipping plane 
         *
         * @return The distance of the near clipping plane 
         */
        virtual SceneSpaceType NearClip(void) const = 0;

        /** 
         * Answer the position of the camera in world coordinates 
         *
         * @return The position of the camera in world coordinates 
         */
        virtual const math::Point<SceneSpaceType, 3>& Position(void) const = 0;

        /** 
         * Answer the type of stereo projection 
         *
         * @return The type of stereo projection 
         */
        virtual ProjectionType Projection(void) const = 0;

        /** Sets all parameters to their default values. */
        virtual void Reset(void) = 0;

        /** resets the clip tile rectangle to the whole virtual view size */
        virtual void ResetTileRect(void) = 0;

        /** 
         * Answer the normalised right vector of the camera. This vector
         * depends on the coordinate system type of the camera.
         *
         * @return The normalised right vector of the camera. 
         */
        virtual const math::Vector<SceneSpaceType, 3>& Right(void) const = 0;

        /**
         * Serialise the object to 'serialiser'. The caller must ensure that
         * the Serialiser is in an acceptable state to serialise to.
         *
         * The camera parameter limits are NOT serialised!
         *
         * @param serialiser The Serialiser to serialise the object to.
         *
         * @throws Exception Implementing classes may throw exceptions to 
         *                   indicate an error or pass through exceptions thrown
         *                   by the Serialiser.
         */
        virtual void Serialise(Serialiser& serialiser) const;

        /**
         * Sets the aperture angle along the y-axis.
         *
         * @param The aperture angle in radians.
         */
        virtual void SetApertureAngle(math::AngleDeg apertureAngle) = 0;

        /**
         * Sets the autofocus offset
         *
         * @param offset The new autofocus offset
         */
        virtual void SetAutoFocusOffset(SceneSpaceType offset) = 0;

        /**
         * Sets the clipping distances.
         *
         * @param nearClip the distance to the near clipping plane.
         * @param farClip the distance to the far clipping plane.
         */
        virtual void SetClip(SceneSpaceType nearClip, SceneSpaceType farClip) = 0;

        /**
         * Sets the coordinate system type the camera is used in.
         *
         * @param coordSysType The new coordinate system type to use.
         */
        virtual void SetCoordSystemType(math::CoordSystemType coordSysType) = 0;

        /**
         * Sets the eye for stereo projection.
         *
         * @param eye The eye for stereo projection.
         */
        virtual void SetEye(StereoEye eye) = 0;

        /**
         * Sets the far clipping distance.
         *
         * @param farClip the distance to the far clipping plane.
         */
        virtual void SetFarClip(SceneSpaceType farClip) = 0;

        /**
         * Sets the focal distance. A value of zero will activate auto-focus
         * to the look at point.
         *
         * @param focalDistance The focal distance.
         */
        virtual void SetFocalDistance(SceneSpaceType focalDistance) = 0;

        /**
         * Sets the limits object limiting the values of all values. This also
         * implicitly calls 'ApplyLimits'.
         *
         * @param limits The new limits object.
         */
        virtual void SetLimits(const SmartPtr<CameraParameterLimits>& limits) = 0;

        /**
         * Sets the look-at-point of the camera in world coordinates.
         *
         * @param lookAt The look-at-point of the camera in world coordinates.
         */
        virtual void SetLookAt(const math::Point<SceneSpaceType, 3>& lookAt) = 0;

        /**
         * Sets the near clipping distance.
         *
         * @param nearClip the distance to the near clipping plane.
         */        
        virtual void SetNearClip(SceneSpaceType nearClip) = 0;

        /**
         * Sets the position of the camera in world coordinates.
         *
         * @param position The position of the camera in world coordinates.
         */
        virtual void SetPosition(const math::Point<SceneSpaceType, 3>& position) = 0;

        /**
         * Sets the projection type used.
         *
         * @param projectionType The projection type used.
         */
        virtual void SetProjection(ProjectionType projectionType) = 0;

        /**
         * Sets the stereo disparity.
         *
         * @param stereoDisparity The stereo disparity.
         */
        virtual void SetStereoDisparity(SceneSpaceType stereoDisparity) = 0;

        /**
         * Sets all stereo parameters.
         *
         * @param stereoDisparity The stereo disparity.
         * @param eye The eye for stereo projection.
         * @param focalDistance The focal distance.
         */
        virtual void SetStereoParameters(SceneSpaceType stereoDisparity, 
            StereoEye eye, SceneSpaceType focalDistance) = 0;

        /**
         * Sets the selected clip tile rectangle of the virtual view. Also see
         * 'SetVirtualViewSize' for further information.
         *
         * @param tileRect The selected clip tile rectangle of the virtual view
         */
        virtual void SetTileRect(const math::Rectangle<ImageSpaceType>& tileRect) = 0;

        /**
         * Sets the up vector of the camera in world coordinates.
         *
         * @param up The up vector of the camera in world coordinates.
         */
        virtual void SetUp(const math::Vector<SceneSpaceType, 3>& up) = 0;

        /**
         * Sets the view position and direction parameters of the camera in
         * world coodirnates. 'position' is the most important value, so if
         * the limits are violated or if the vectors do not construct an
         * orthogonal koordinate system, 'lookAt' and 'up' may be changed.
         * This is true for all 'Set'-Methods.
         *
         * @param position The position of the camera in world coordinates.
         * @param lookAt The look-at-point of the camera in world coordinates.
         * @param up The up vector of the camera in world coordinates.
         */
        virtual void SetView(const math::Point<SceneSpaceType, 3>& position, 
            const math::Point<SceneSpaceType, 3>& lookAt, 
            const math::Vector<SceneSpaceType, 3>& up) = 0;

        /**
         * Sets the size of the full virtual view. If the selected clip tile 
         * rectangle covered the whole virtual view the clip tile rectangle 
         * will also be changed that it still covers the whole virtual view. 
         * Otherwise the clip tile rectangle will not be changed.
         *
         * @param viewSize The new size of the full virtual view.
         */
        virtual void SetVirtualViewSize(const math::Dimension<ImageSpaceType, 2>& viewSize) = 0;

        /**
         * Sets the size of the full virtual view. If the selected clip tile 
         * rectangle covered the whole virtual view the clip tile rectangle 
         * will also be changed that it still covers the whole virtual view. 
         * Otherwise the clip tile rectangle will not be changed.
         *
         * @param viewWidth The new width of the full virtual view.
         * @param viewHeight The new height of the full virtual view.
         */
        inline void SetVirtualViewSize(ImageSpaceType viewWidth,
                ImageSpaceType viewHeight) {
            math::Dimension<ImageSpaceType, 2> viewSize(viewWidth, viewHeight);
            this->SetVirtualViewSize(viewSize);
        }

        /** 
         * Answer the eye disparity value for stereo images.
         *
         * @return The eye disparity value for stereo images.
         */
        inline SceneSpaceType StereoDisparity(void) const {
            return static_cast<SceneSpaceType>(2) 
                * this->HalfStereoDisparity();
        }

        /** 
         * Answer the synchronisation number used to update camera objects 
         * using this parameters object.
         *
         * @return The synchronisation number.
         */
        virtual unsigned int SyncNumber(void) const = 0;

        /** 
         * Answer the selected clip tile rectangle of the virtual view. (E. g.
         * this should be used as rendering viewport).
         *
         * @return The selected clip tile rectangle 
         */
        virtual const math::Rectangle<ImageSpaceType>& TileRect(void) const = 0;

        /** 
         * Answer the normalised up vector of the camera. The vector 
         * (lookAt - position) and this vector must not be parallel.
         *
         * @return The normalised up vector of the camera. 
         */
        virtual const math::Vector<SceneSpaceType, 3>& Up(void) const = 0;

        /** 
         * Answer the size of the full virtual view.
         *
         * @return The size of the full virtual view.
         */
        virtual const math::Dimension<ImageSpaceType, 2>& VirtualViewSize(void) const = 0;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParameters& operator=(const CameraParameters& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator==(const CameraParameters& rhs) const;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMETERS_H_INCLUDED */
