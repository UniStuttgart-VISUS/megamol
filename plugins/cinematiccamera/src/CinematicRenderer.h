/*
* CinematicRenderer.h
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATICCAMERA_CINEMATICRENDERER_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_CINEMATICRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/BoundingBoxes.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"

#include "mmcore/utility/SDFFont.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FilePathParam.h"

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/View3D.h"

#include "vislib/Trace.h"
#include "vislib/String.h"
#include "vislib/Array.h"
#include "vislib/memutils.h"
#include "vislib/StringSerialiser.h"

#include "vislib/math/Point.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"

#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Thread.h"

#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/CameraParamsStore.h"

#include "CallCinematicCamera.h"
#include "ReplacementRenderer.h"
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

		protected:

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

			/**
			* The get extents callback. The module should set the members of
			* 'call' to tell the caller the extents of its data (bounding boxes
			* and times).
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool GetExtents(megamol::core::view::CallRender3D& call);

			/**
			* The render callback.
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool Render(megamol::core::view::CallRender3D& call);

            /** The mouse button pressed/released callback. */
            virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) override;

            /** The mouse movement callback. */
            virtual bool OnMouseMove(double x, double y) override;

            //virtual bool OnKey(megamol::core::view::Key key, megamol::core::view::KeyAction action, megamol::core::view::Modifiers mods) override;

            //virtual bool OnChar(unsigned int codePoint) override;

            //virtual bool OnMouseScroll(double dx, double dy) override;

		private:

            /**********************************************************************
            * variables
            **********************************************************************/

            // font rendering
            megamol::core::utility::SDFFont theFont;

            unsigned int                     interpolSteps;
            unsigned int                     toggleManipulator;
            bool                             manipOutsideModel;
            bool                             showHelpText;

            KeyframeManipulator              manipulator;
            bool                             manipulatorGrabbed;
            bool                             showMode;

            vislib::graphics::gl::FramebufferObject fbo;

            /** The render to texture shader */
            vislib::graphics::gl::GLSLShader textureShader;

            /*** INPUT ********************************************************/

            /** The mouse coordinates */
            float                            mouseX;
            float                            mouseY;

            /**********************************************************************
            * callback stuff
            **********************************************************************/

			/** The renderer caller slot */
			core::CallerSlot rendererCallerSlot;

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
            core::param::ParamSlot toggleManipOusideBboxParam;

			bool isSelecting;

		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CINEMATICRENDERER_H_INCLUDED */
