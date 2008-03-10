/*
 * ObservableCameraParams.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OBSERVABLECAMERAPARAMS_H_INCLUDED
#define VISLIB_OBSERVABLECAMERAPARAMS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CameraParameterObserver.h"
#include "vislib/CameraParamsOverride.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {
namespace graphics {


    /**
     * This class implements a specialised CameraParamsStore that fires events
     * every time it was changed.
     *
     * Note: THE CURRENT IMPLEMENTATION IS NOT THREAD-SAFE!
     *
     * Note: This is not a CameraParamsOverride! It only inherits from this class
     * because I am so lazy. The observer is implemented via a loop-through
     * in the setters. Therefore, it is not safe against manipulations using
     * static_casts.
     */
    class ObservableCameraParams : public CameraParamsOverride {

    public:

        /**
         * Ctor. 
         *
         * @param base The underlying camera parameters that are observed.
         */
        ObservableCameraParams(SmartPtr<CameraParameters>& base);

        /**
         * Copy ctor. 
         *
         * @param rhs The right hand side operand.
         */
        ObservableCameraParams(const ObservableCameraParams& rhs);

        /** Dtor. */
        virtual ~ObservableCameraParams(void);

        /**
         * Register a new observer that is informed about camera parameter 
         * changes.
         *
         * NOTE: The object does NOT ownership of the object designated by
         * 'observer'. The caller is responsible that 'observer' exists as
         * long as it is registered.
         *
         * @param observer The observer to be registered. This must not be
         *                 NULL.
         */
        virtual void AddCameraParameterObserver(
            CameraParameterObserver *observer);

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
         * Begin a batch interaction that accumulates all changes to the camera
         * parameters instead of firing events directly. All change events 
         * that the object receives until a call to EndBatchInteraction will be
         * accumulated.
         */
        virtual void BeginBatchInteraction(void);

        /**
         * Ends a batch interaction and fires all pending events.
         */
        virtual void EndBatchInteraction(void);

        /**
         * Removes the observer 'observer' from the list ob registered 
         * camera parameter observers. It is safe to remove non-registered
         * observers.
         *
         * @param observer The observer to be removed. This must not be NULL.
         */
        virtual void RemoveCameraParameterObserver(
            CameraParameterObserver *observer);

        /** 
         * Sets all parameters to their default values. 
         */
        virtual void Reset(void);

        /** 
         * Resets the clip tile rectangle to the whole virtual view size.
         */
        virtual void ResetTileRect(void);

       /**
         * Sets the aperture angle along the y-axis.
         *
         * @param The aperture angle in radians.
         */
        virtual void SetApertureAngle(math::AngleDeg apertureAngle);

        /**
         * Sets the clipping distances.
         *
         * @param nearClip the distance to the near clipping plane.
         * @param farClip the distance to the far clipping plane.
         */
        virtual void SetClip(SceneSpaceType nearClip, SceneSpaceType farClip);

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
         * Sets the focal distance.
         *
         * @param focalDistance The focal distance.
         */
        virtual void SetFocalDistance(SceneSpaceType focalDistance);

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
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        ObservableCameraParams& operator =(
            const ObservableCameraParams& rhs);

        /**
         * Test for equality.
         *
         * Note: The observer state and the dirty flags are not part of the 
         * state that is compared.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if the objects are equal with respect to the equality
         *         conditions of CameraParamsStore, false otherwise.
         */
        bool operator ==(const ObservableCameraParams& rhs) const;

    protected:

        /** Direct superclass. */
        typedef CameraParamsOverride Super;

        /** Set all parameters dirty. */
        static const UINT32 DIRTY_ALL;

        /** The dirty flag for the aperture angle. */
        static const UINT32 DIRTY_APERTUREANGLE;

        /** The dirty flag for the stereo eye. */
        static const UINT32 DIRTY_EYE;

        /** The dirty flag for the far clipping plane. */
        static const UINT32 DIRTY_FARCLIP;

        /** The dirty flag for the focal distance. */
        static const UINT32 DIRTY_FOCALDISTANCE;

        /** The dirty flag for the parameter limits. */
        static const UINT32 DIRTY_LIMITS;

        /** The dirty flag for the look-at point. */
        static const UINT32 DIRTY_LOOKAT;

        /** The dirty flag for the near clipping plane. */
        static const UINT32 DIRTY_NEARCLIP;

        /** The dirty flag for the camera position. */
        static const UINT32 DIRTY_POSITION;
        
        /** The dirty flag for the stereo/mono projection type. */
        static const UINT32 DIRTY_PROJECTION;

        /** The dirty flag for the stereo disparity. */
        static const UINT32 DIRTY_DISPARITY;

        /** The dirty flag for the tile rectangle. */
        static const UINT32 DIRTY_TILERECT;

        /** The dirty flag for the up vector. */
        static const UINT32 DIRTY_UP;
        
        /** The dirty flag for the virtual view size. */
        static const UINT32 DIRTY_VIRTUALVIEW;

        /**
         * Inform all registered observers about a change of the fields marked
         * dirty in the 'which' bitfield.
         *
         * If either 'isBatchInteraction' or 'isSuspendFire' are set, no event
         * will be fired, but 'which' will be added to the 'dirtyFields' member.
         *
         * If an event is actually fired, it is reset in the 'dirtyFields' 
         * member.
         *
         * If 'andAllDirty' is set, all dirty fields are also fired, even if 
         * they are not set in 'which'. The 'andAllDirty' flag has no effect
         * if firing of events is suspended internally or by the user.
         *
         * @param which       A bitfield of DIRTY_* that defines the events to 
         *                    be fired. Defaults to DIRTY_ALL.
         * @param andAllDirty If set true, events for all fields that are marked
         *                    in the 'dirtyFields' member are fired, too. If set
         *                    false, only fields set in 'which' are fired. 
         *                    Defaults to true.
         */
        void fireChanged(const UINT32 which = DIRTY_ALL, 
            const bool andAllDirty = true);

        /**
         * Resumes firing of events. No events are actually fired.
         */
        inline void resumeFire(void) {
            this->isSuspendFire = false;
        }

        /**
         * Suspends firing of events.
         */
        inline void suspendFire(void) {
            this->isSuspendFire = true;
        }

        /**
         * This bitmask accumulates the dirty fields that are updated within a
         * BeginBatchInteraction/EndBatchInteraction block or while directly 
         * firing events is internally suspended.
         */
        UINT32 dirtyFields;

        /** The list of registered CameraParameterObservers. */
        SingleLinkedList<CameraParameterObserver *> camParamObservers;

        /** Enables or disables accumulation of interaction operations. */
        bool isBatchInteraction;

    private:

        /**
         * Disables firing events directly. This is for internal use only: If
         * the flag is set, firing multiple events because of recursion or 
         * internal use of publicly visible methods is prevented.
         */
        bool isSuspendFire;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_OBSERVABLECAMERAPARAMS_H_INCLUDED */
