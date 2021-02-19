/*
 * Diagram2DRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * All Rights reserved.
 */

#ifndef MEGAMOLCORE_DIAGRAM2DRENDERER_H_INCLUDED
#define MEGAMOLCORE_DIAGRAM2DRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer2DModule.h"
#include "Diagram2DCall.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace protein {
	/**
	 * Protein Renderer class
	 */
	class Diagram2DRenderer : public megamol::core::view::Renderer2DModule {
    public:
		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void) {
			return "Diagram2DRenderer";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) {
			return "Offers diagram renderings.";
		}

		/**
		 * Answers whether this module is available on the current system.
		 *
		 * @return 'true' if the module is available, 'false' otherwise.
		 */
		static bool IsAvailable(void) {
			return true;
		}

        /** ctor */
        Diagram2DRenderer(void);

        /** dtor */
        ~Diagram2DRenderer(void);

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
         * Callback for mouse events (move, press, and release)
         *
         * @param x The x coordinate of the mouse in world space
         * @param y The y coordinate of the mouse in world space
         * @param flags The mouse flags
         */
        virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

	private:

		/**********************************************************************
		 * 'render'-functions
		 **********************************************************************/

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents( megamol::core::view::CallRender2DGL& call);

		/**
		* The Open GL Render callback.
		*
		* @param call The calling call.
		* @return The return value of the function.
		*/
        virtual bool Render( megamol::core::view::CallRender2DGL& call);

        /**
         * Generate the diagram textures and FBOs.
         */
        void generateDiagramTextures();

        /**
         * Clears the diagram textures.
         */
        void clearDiagram();

        /*
         * Refresh all parameters.
         */
        void parameterRefresh();

		/**********************************************************************
		 * variables
		 **********************************************************************/

		/** caller slot */
		core::CallerSlot dataCallerSlot;

		/** texture resolution parameter */
		megamol::core::param::ParamSlot resolutionParam;
        /** plot color parameter */
		megamol::core::param::ParamSlot plotColorParam;
        /** clear diagram parameter */
		megamol::core::param::ParamSlot clearDiagramParam;

        /** the mouse position */
        vislib::math::Vector<float, 3> mousePos;

        /** the current fbo (for pingpong rendering) */
        unsigned int currentFbo;
        /** the diagram framebuffer objects */
        vislib::graphics::gl::FramebufferObject fbo[2];

        /** the last data point */
        vislib::math::Vector<float, 2> oldDataPoint;

        /** the plot color */
        vislib::math::Vector<float, 3> plotColor;

        /** the last label area */
        vislib::math::Vector<float, 3> labelSpace;
    };

}
}

#endif // MEGAMOLCORE_DIAGRAM2DRENDERER_H_INCLUDED
