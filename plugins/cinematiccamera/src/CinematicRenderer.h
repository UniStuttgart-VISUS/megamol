/*
* CinematicRenderer.h
*
*/

#ifndef MEGAMOL_CINEMATICCAMERA_CINEMATICRENDERER_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_CINEMATICRENDERER_H_INCLUDED
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

#include "vislib/graphics/gl/GLSLShader.h"
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

#include "KeyframeManipulator.h"

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

            /**********************************************************************
            * variables
            **********************************************************************/

            // font rendering
#ifdef USE_SIMPLE_FONT
            vislib::graphics::gl::SimpleFont  theFont;
#else
            vislib::graphics::gl::OutlineFont theFont;
#endif
            unsigned int                     interpolSteps;
            bool                             toggleManipulator;
            bool                             showHelpText;
            bool                             toggleModelBBox;
            KeyframeManipulator              manipulator;
            vislib::graphics::gl::FramebufferObject fbo;
            vislib::math::Cuboid<float>      ocBbox;
            /** The render to texture */
            //vislib::graphics::gl::GLSLShader textureShader;

            /**********************************************************************
            * functions
            **********************************************************************/

            /** */
            void drawBoundingBox(void);

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
            /**  */
            core::param::ParamSlot toggleManipulateParam;
            /**  */
            core::param::ParamSlot toggleHelpTextParam;
            /**  */
            core::param::ParamSlot toggleModelBBoxParam;
		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CINEMATICRENDERER_H_INCLUDED */
