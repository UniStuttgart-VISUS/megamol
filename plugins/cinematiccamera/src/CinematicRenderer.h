/*
* CinematicRenderer.h
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_CinematicRenderer_H_INCLUDED
#define MEGAMOLCORE_CinematicRenderer_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/BoundingBoxes.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/String.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/graphics/gl/Verdana.inc"


namespace megamol {
	namespace cinematiccamera {
		
		/**
		* A renderer that passes the render call to another renderer
		*/
		
		class CinematicRenderer : public core::view::Renderer3DModule {
		public:

			/**
			* Gets the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "CinematicRenderer";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Renderer that passes the render call to another renderer";
			}

			/**
			* Gets whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}

			/**
			* Disallow usage in quickstarts
			*
			* @return false
			*/
			static bool SupportQuickstart(void) {
				return false;
			}

			/** Ctor. */
			CinematicRenderer(void);

			/** Dtor. */
			virtual ~CinematicRenderer(void);

			/**
			* Implementation of 'Create'.
			*
			* @return 'true' on success, 'false' otherwise.
			*/
			virtual bool create(void);

			/**
			* Implementation of 'Release'.
			*/
			virtual void release(void);

		protected:

  			/**
			* The get capabilities callback. The module should set the members
			* of 'call' to tell the caller its capabilities.
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool GetCapabilities(core::Call& call);

			/**
			* The get extents callback. The module should set the members of
			* 'call' to tell the caller the extents of its data (bounding boxes
			* and times).
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool GetExtents(core::Call& call);

			/**
			* The render callback.
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool Render(core::Call& call);

            /**
            * Callback for mouse events (move, press, and release)
            *
            * @param x The x coordinate of the mouse in world space
            * @param y The y coordinate of the mouse in world space
            * @param flags The mouse flags
            */
            virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

		private:

            // enumeration of manipulator types
            enum manipulatorType {
                NONE          = 0,
                CAM_X_AXIS    = 1,
                CAM_Y_AXIS    = 2,
                CAM_Z_AXIS    = 3,
                CAM_POS       = 4,
                CAM_UP        = 5,
                LOOKAT_X_AXIS = 6,
                LOOKAT_Y_AXIS = 7,
                LOOKAT_Z_AXIS = 8
            };

            // enumeration of color types
            enum colType {
                COL_SPLINE          = 0,
                COL_KEYFRAME        = 1,
                COL_SELECT_KEYFRAME = 2,
                COL_SELECT_LOOKAT   = 3,
                COL_SELECT_UP       = 4,
                COL_SELECT_X_AXIS   = 5,
                COL_SELECT_Y_AXIS   = 6,
                COL_SELECT_Z_AXIS   = 7
            };

            /**********************************************************************
            * functions
            **********************************************************************/

            /**
            * Check if mouse position hits point (as vector) in world space 
            * coordinates within some offset.
            *
            * @param x 
            * @param y
            * @param o (=origin)
            * @param m (=manipulator)
            *
            * @return True if point is hit.
            */
            bool processPointHit(float x, float y, vislib::math::Point<GLfloat, 3> camPos, vislib::math::Point<GLfloat, 3> manipPos, manipulatorType t);

            /** Render 2D circle facing to the camera position */
            void renderCircle2D(float radius, unsigned int subdiv, vislib::math::Point<GLfloat, 3> camPos, vislib::math::Point<GLfloat, 3> centerPos, vislib::math::Vector<float, 3> col);

            /**********************************************************************
            * variables
            **********************************************************************/

            struct manipulator {
                bool                           active;
                manipulatorType                type;
                vislib::math::Vector<float, 3> lastMouse;
                vislib::math::Vector<float, 3> ssKeyframePos;
                vislib::math::Vector<float, 3> ssManipulatorPos;
            };

            // font rendering
#ifdef USE_SIMPLE_FONT
            vislib::graphics::gl::SimpleFont  theFont;
#else
            vislib::graphics::gl::OutlineFont theFont;
#endif

            vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix;
            vislib::Array<vislib::math::Vector<float, 3> > colors;
            vislib::math::Rectangle<int>  viewport;
            unsigned int                  interpolSteps;
            bool                          toggleManipulator;
            manipulator                   currentManipulator;
            float                         maxAnimTime;
            vislib::math::Point<float, 3> bboxCenter;

            /**********************************************************************
            * callback stuff
            **********************************************************************/

			/** The renderer caller slot */
			core::CallerSlot slaveRendererSlot;

			/** The keyframe keeper caller slot */
			core::CallerSlot keyframeKeeperSlot;

            /**********************************************************************
            * parameters
            **********************************************************************/
			
            /** Amount of interpolation steps between keyframes */
            core::param::ParamSlot stepsParam;
            /** Param to toggle to manipulation mode of position or camera lookup */
            core::param::ParamSlot toggleManipulateParam;
		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CinematicRenderer_H_INCLUDED */
