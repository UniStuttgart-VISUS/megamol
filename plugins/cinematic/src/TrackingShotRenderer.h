/*
* TrackingShotRenderer.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_TRACKINGSHOTRENDERER_H_INCLUDED
#define MEGAMOL_CINEMATIC_TRACKINGSHOTRENDERER_H_INCLUDED

#include "Cinematic/Cinematic.h"

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
#include "mmcore/view/Renderer3DModule_2.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/View3D.h"
#include "mmcore/view/Input.h"

#include "vislib/Trace.h"
#include "vislib/String.h"
#include "vislib/Array.h"
#include "vislib/memutils.h"
#include "vislib/StringSerialiser.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Thread.h"

#include "CallKeyframeKeeper.h"
#include "ReplacementRenderer.h"
#include "KeyframeManipulators.h"
#include "CinematicUtils.h"


namespace megamol {
namespace cinematic {
		
	/**
	* Tracking shot rendering.
	*/
	class TrackingShotRenderer : public core::view::Renderer3DModule_2 {
	public:

		/**
		* Gets the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "TrackingShotRenderer";
		}

		/**
		* Gets a human readable description of the module.
		*
		* @return A human readable description of the module.
		*/
		static const char *Description(void) {
			return "Renders the tracking shot and passes the render call to another renderer.";
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
		TrackingShotRenderer(void);

		/** Dtor. */
		virtual ~TrackingShotRenderer(void);

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
		virtual bool GetExtents(megamol::core::view::CallRender3D_2& call);

        /**
         * Method that gets called before the rendering is started for all changed modules
         *
         * @param call The rendering call that contains the camera
         */
        void PreRender(megamol::core::view::CallRender3D_2& call);

		/**
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		virtual bool Render(megamol::core::view::CallRender3D_2& call);

        /** 
        * The mouse button pressed/released callback. 
        */
        virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) final override;

        /** 
        * The mouse movement callback. 
        */
        virtual bool OnMouseMove(double x, double y) final override;

	private:

        /**********************************************************************
        * variables
        **********************************************************************/

        KeyframeManipulators                    manipulators;
        CinematicUtils                          utils;
        vislib::graphics::gl::FramebufferObject fbo;
        float                                   mouseX;
        float                                   mouseY;
        GLuint                                  texture;
        bool                                    manipulatorGrabbed;
        unsigned int                            interpolSteps;
        bool                                    showHelpText;
        float                                   lineWidth;

        /**********************************************************************
        * callbacks
        **********************************************************************/

		core::CallerSlot keyframeKeeperSlot;

        /**********************************************************************
        * parameters
        **********************************************************************/
			
        core::param::ParamSlot stepsParam; // Amount of interpolation steps between keyframes
        core::param::ParamSlot toggleHelpTextParam;

	};

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_TRACKINGSHOTRENDERER_H_INCLUDED