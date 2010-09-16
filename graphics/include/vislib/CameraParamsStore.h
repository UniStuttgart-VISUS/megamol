/*
 * CameraParamsStore.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAPARAMSSTORE_H_INCLUDED
#define VISLIB_CAMERAPARAMSSTORE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameters.h"


namespace vislib {
namespace graphics {


    /**
     * Implementation of 'CameraParameters' storing all values.
     */
    class CameraParamsStore : public CameraParameters {
    public:

        /** Ctor. */
        CameraParamsStore(void);

        /**
         * Copy ctor. 
         *
         * @param rhs The right hand side operand.
         */
        CameraParamsStore(const CameraParamsStore& rhs);

        /**
         * Copy ctor. 
         *
         * @param rhs The right hand side operand.
         */
        CameraParamsStore(const CameraParameters& rhs);

        /** Dtor. */
        virtual ~CameraParamsStore(void);

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
        virtual void ApplyLimits(void);

        /**
         * Answers the auto focus offset
         *
         * @return The auto focus offset
         */
        virtual SceneSpaceType AutoFocusOffset(void) const;

        /**
         * Answer the coordinate system type of the camera.
         *
         * @return the coordinate system type of the camera.
         */
        virtual math::CoordSystemType CoordSystemType(void) const;

        /**
         * Answer the eye for stereo projections.
         *
         * @return The eye for stereo projections.
         */
        virtual StereoEye Eye(void) const;

        /**
         * Calculates and returns the real eye looking direction taking stereo
         * projection mode and stereo disparity into account.
         *
         * @return The real eye looking direction.
         */
        virtual math::Vector<SceneSpaceType, 3> EyeDirection(void) const;

        /**
         * Calculates and returns the real eye looking direction taking stereo
         * projection mode and stereo disparity into account.
         *
         * @return The real eye looking direction.
         */
        virtual math::Vector<SceneSpaceType, 3> EyeUpVector(void) const;

        /**
         * Calculates and returns the real eye looking direction taking stereo
         * projection mode and stereo disparity into account. This vector
         * depends on the coordinate system type of the camera.
         *
         * @return The real eye looking direction.
         */
        virtual math::Vector<SceneSpaceType, 3> EyeRightVector(void) const;

        /**
         * Calculates and returns the real eye position taking stereo 
         * projection mode and stereo disparity into account. This vector
         * depends on the coordinate system type of the camera.
         *
         * @return The real eye position.
         */
        virtual math::Point<SceneSpaceType, 3> EyePosition(void) const;

        /** 
         * Answer the distance of the far clipping plane 
         *
         * @return The distance of the far clipping plane 
         */
        virtual SceneSpaceType FarClip(void) const;

        /** 
         * Answer the focal distance for stereo images 
         *
         * @return The focal distance for stereo images 
         */
        virtual SceneSpaceType FocalDistance(bool autofocus = true) const;

        /** 
         * Answer the normalised front vector of the camera. 
         *
         * @return The normalised front vector of the camera. 
         */
        virtual const math::Vector<SceneSpaceType, 3>& Front(void) const;

        /**
         * Answer the half aperture angle in radians.
         *
         * @return The half aperture angle in radians.
         */
        virtual math::AngleRad HalfApertureAngle(void) const;

        /**
         * Answer the half stereo disparity.
         *
         * @return The half stereo disparity.
         */
        virtual SceneSpaceType HalfStereoDisparity(void) const;

        /**
         * Answer whether this camera parameters object is similar with the
         * specified one. Similarity is given if the objects are equal or are
         * based on equal objects.
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if both objects are similar, 'false' otherwise.
         */
        virtual bool IsSimilar(const SmartPtr<CameraParameters> rhs) const;

        /**
         * Answer the limits object used by this object.
         *
         * @return The limits object.
         */
        virtual SmartPtr<CameraParameterLimits> Limits(void) const;

        /** 
         * Asnwer the look-at-point of the camera in world coordinates 
         *
         * @return The look-at-point of the camera in world coordinates 
         */
        virtual const math::Point<SceneSpaceType, 3>& LookAt(void) const;

        /** 
         * Answer the distance of the near clipping plane 
         *
         * @return The distance of the near clipping plane 
         */
        virtual SceneSpaceType NearClip(void) const;

        /** 
         * Answer the position of the camera in world coordinates 
         *
         * @return The position of the camera in world coordinates 
         */
        virtual const math::Point<SceneSpaceType, 3>& Position(void) const;

        /** 
         * Answer the type of stereo projection 
         *
         * @return The type of stereo projection 
         */
        virtual ProjectionType Projection(void) const;

        /** Sets all parameters to their default values. */
        virtual void Reset(void);

        /** resets the clip tile rectangle to the whole virtual view size */
        virtual void ResetTileRect(void);

        /** 
         * Answer the normalised right vector of the camera. This vector
         * depends on the coordinate system type of the camera.
         *
         * @return The normalised right vector of the camera. 
         */
        virtual const math::Vector<SceneSpaceType, 3>& Right(void) const;

        /**
         * Sets the aperture angle along the y-axis.
         *
         * @param The aperture angle in radians.
         */
        virtual void SetApertureAngle(math::AngleDeg apertureAngle);

        /**
         * Sets the autofocus offset
         *
         * @param offset The new autofocus offset
         */
        virtual void SetAutoFocusOffset(SceneSpaceType offset);

        /**
         * Sets the clipping distances.
         *
         * @param nearClip the distance to the near clipping plane.
         * @param farClip the distance to the far clipping plane.
         */
        virtual void SetClip(SceneSpaceType nearClip, SceneSpaceType farClip);

        /**
         * Sets the coordinate system type the camera is used in.
         *
         * @param coordSysType The new coordinate system type to use.
         */
        virtual void SetCoordSystemType(math::CoordSystemType coordSysType);

        /**
         * Sets the eye for stereo projection.
         *
         * @param eye The eye for stereo projection.
         */
        virtual void SetEye(StereoEye eye);

        /**
         * Sets the far clipping distance.
         *
         * @param farClip the distance to the far clipping plane.
         */
        virtual void SetFarClip(SceneSpaceType farClip);

        /**
         * Sets the focal distance. A value of zero will activate auto-focus
         * to the look at point.
         *
         * @param focalDistance The focal distance.
         */
        virtual void SetFocalDistance(SceneSpaceType focalDistance);

        /**
         * Sets the limits object limiting the values of all values. This also
         * implicitly calls 'ApplyLimits'.
         *
         * @param limits The new limits object.
         */
        virtual void SetLimits(const SmartPtr<CameraParameterLimits>& limits);

        /**
         * Sets the look-at-point of the camera in world coordinates.
         *
         * @param lookAt The look-at-point of the camera in world coordinates.
         */
        virtual void SetLookAt(const math::Point<SceneSpaceType, 3>& lookAt);

        /**
         * Sets the near clipping distance.
         *
         * @param nearClip the distance to the near clipping plane.
         */        
        virtual void SetNearClip(SceneSpaceType nearClip);

        /**
         * Sets the position of the camera in world coordinates.
         *
         * @param position The position of the camera in world coordinates.
         */
        virtual void SetPosition(
            const math::Point<SceneSpaceType, 3>& position);

        /**
         * Sets the projection type used.
         *
         * @param projectionType The projection type used.
         */
        virtual void SetProjection(ProjectionType projectionType);

        /**
         * Sets the stereo disparity.
         *
         * @param stereoDisparity The stereo disparity.
         */
        virtual void SetStereoDisparity(SceneSpaceType stereoDisparity);

        /**
         * Sets all stereo parameters.
         *
         * @param stereoDisparity The stereo disparity.
         * @param eye The eye for stereo projection.
         * @param focalDistance The focal distance.
         */
        virtual void SetStereoParameters(SceneSpaceType stereoDisparity, 
            StereoEye eye, SceneSpaceType focalDistance);

        /**
         * Sets the selected clip tile rectangle of the virtual view. Also see
         * 'SetVirtualViewSize' for further information.
         *
         * @param tileRect The selected clip tile rectangle of the virtual view
         */
        virtual void SetTileRect(
            const math::Rectangle<ImageSpaceType>& tileRect);

        /**
         * Sets the up vector of the camera in world coordinates.
         *
         * @param up The up vector of the camera in world coordinates.
         */
        virtual void SetUp(const math::Vector<SceneSpaceType, 3>& up);

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
            const math::Vector<SceneSpaceType, 3>& up);

        /**
         * Sets the size of the full virtual view. If the selected clip tile 
         * rectangle covered the whole virtual view the clip tile rectangle 
         * will also be changed that it still covers the whole virtual view. 
         * Otherwise the clip tile rectangle will not be changed.
         *
         * @param viewSize The new size of the full virtual view.
         */
        virtual void SetVirtualViewSize(
            const math::Dimension<ImageSpaceType, 2>& viewSize);

        /** 
         * Answer the synchronisation number used to update camera objects 
         * using this parameters object.
         *
         * @return The synchronisation number.
         */
        virtual unsigned int SyncNumber(void) const;

        /** 
         * Answer the selected clip tile rectangle of the virtual view. (E. g.
         * this should be used as rendering viewport).
         *
         * @return The selected clip tile rectangle 
         */
        virtual const math::Rectangle<ImageSpaceType>& TileRect(void) const;

        /** 
         * Answer the normalised up vector of the camera. The vector 
         * (lookAt - position) and this vector must not be parallel.
         *
         * @return The normalised up vector of the camera. 
         */
        virtual const math::Vector<SceneSpaceType, 3>& Up(void) const;

        /** 
         * Answer the size of the full virtual view.
         *
         * @return The size of the full virtual view.
         */
        virtual const math::Dimension<ImageSpaceType, 2>& VirtualViewSize(void) const;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsStore& operator=(const CameraParamsStore& rhs);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsStore& operator=(const CameraParameters& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator==(const CameraParamsStore& rhs) const;

    private:

        /** The auto focus offset */
        SceneSpaceType autoFocusOffset;

        /** The coordinate system type of the camera */
        math::CoordSystemType coordSysType;

        /** eye for stereo projections */
        StereoEye eye;

        /** distance of the far clipping plane */
        SceneSpaceType farClip;

        /** focal distance for stereo images */
        SceneSpaceType focalDistance;

        /** Normalised front vector of the camera. */
        math::Vector<SceneSpaceType, 3> front;

        /** 
         * half aperture Angle in radians of the camera along the y axis on the
         * virtual view.
         */
        math::AngleRad halfApertureAngle;

        /** half eye disparity value for stereo images */
        SceneSpaceType halfStereoDisparity;

        /** the limiting values used for this camera parameter object */
        vislib::SmartPtr<CameraParameterLimits> limits;

        /** look-at-point of the camera in world coordinates */
        math::Point<SceneSpaceType, 3> lookAt;

        /** distance of the near clipping plane */
        SceneSpaceType nearClip;

        /** position of the camera in world coordinates */
        math::Point<SceneSpaceType, 3> position;

        /** type of stereo projection */
        ProjectionType projectionType;

        /** Normalised right vector of the camera. */
        math::Vector<SceneSpaceType, 3> right;

        /** 
         * the synchronisation number used to update camera objects using this
         * parameters object.
         */
        unsigned int syncNumber;

        /** The selected clip tile rectangle of the virtual view */
        math::Rectangle<ImageSpaceType> tileRect;

        /** 
         * Normalised up vector of the camera. The vector (lookAt - position) 
         * and this vector must not be parallel.
         */
        math::Vector<SceneSpaceType, 3> up;

        /** the size of the full virtual view */
        math::Dimension<ImageSpaceType, 2> virtualViewSize;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAPARAMSSTORE_H_INCLUDED */
