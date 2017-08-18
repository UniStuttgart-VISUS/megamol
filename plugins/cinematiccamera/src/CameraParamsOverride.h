/*
 * CameraParamsOverride.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */
#pragma once

#include "vislib/graphics/CameraParamsOverride.h"


namespace megamol {
namespace cinematiccamera {


    /**
     * Camera parameter override class overriding the eye.
     */
    class CameraParamsOverride : public vislib::graphics::CameraParamsOverride {

    public:

		typedef vislib::graphics::CameraParamsOverride Base;
		typedef vislib::math::Point<vislib::graphics::SceneSpaceType, 3> Point;
		typedef vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> Vector;


        /** Ctor. */
		inline CameraParamsOverride(void) {}

        /** 
         * Ctor. 
         *
         * Note: This is not a copy ctor! This create a new object and sets 
         * 'params' as the camera parameter base object.
         *
         * @param params The base 'CameraParameters' object to use.
         */
		inline CameraParamsOverride(const vislib::SmartPtr<CameraParameters>& params) 
				: Base(params) {
			this->resetOverride();
		}

        /** Dtor. */
        virtual ~CameraParamsOverride(void);

        /**
         * Answer the half aperture angle in radians.
         *
         * @return The half aperture angle in radians.
         */
        virtual vislib::math::AngleRad HalfApertureAngle(void) const;

        /** 
         * Asnwer the look-at-point of the camera in world coordinates 
         *
         * @return The look-at-point of the camera in world coordinates 
         */
		virtual const Point& LookAt(void) const;

		/**
 		 * Answer the position of the camera in world coordinates 
		 *
		 * @return The position of the camera in world coordinates
		 */
		virtual const Point& Position(void) const;

        /**
         * Sets the aperture angle along the y-axis.
         *
         * @param The aperture angle in radians.
         */
        virtual void SetApertureAngle(vislib::math::AngleDeg apertureAngle);

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
        virtual void SetView(const Point& position, const Point& lookAt,
            const Vector& up);

		/**
		* Set Position and LookAt needed for altering keyframes
		*/
		virtual void SetPosition(const Point& position);
		virtual void SetLookAt(const Point& lookAt);

        /** 
         * Answer the normalised up vector of the camera. The vector 
         * (lookAt - position) and this vector must not be parallel.
         *
         * @return The normalised up vector of the camera. 
         */
        virtual const Vector& Up(void) const;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this object.
         */
        CameraParamsOverride& operator=(const CameraParamsOverride& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if all members except the syncNumber are equal, or
         *         'false' if at least one member apart from syncNumber is not
         *         equal.
         */
        bool operator==(const CameraParamsOverride& rhs) const;

    private:

        /**
         * Indicates that a new base object is about to be set.
         *
         * @param params The new base object to be set.
         */
        virtual void preBaseSet(const vislib::SmartPtr<CameraParameters>& params);

        /**
         * Resets the override.
         */
        virtual void resetOverride(void);

		/**
		* half aperture Angle in radians of the camera along the y axis on the
		* virtual view.
		*/
		vislib::math::AngleRad halfApertureAngle;

		Point lookAt;
		Point position;
		Vector up;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

