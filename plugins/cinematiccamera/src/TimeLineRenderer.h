/*
* TimeLineRenderer.h
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATICCAMERA_TIMELINERENDERER_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_TIMELINERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/graphics/gl/Verdana.inc"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "Keyframe.h"
#include "vislib/Array.h"

namespace megamol {
	namespace cinematiccamera {

		/**
		* Mesh-based renderer for bézier curve tubes
		*/
		class TimeLineRenderer : public core::view::Renderer2DModule {
		public:

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "TimeLineRenderer";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Renders the Timeline of keyframes";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}

			/** Ctor. */
			TimeLineRenderer(void);

			/** Dtor. */
			virtual ~TimeLineRenderer(void);

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

			bool LoadTexture(vislib::StringA filename);

			void DrawKeyframeSymbol(Keyframe k, float lineLength, float lineYPos, bool selected);
			
		private:
			
			/**
			* The get extents callback. The module should set the members of
			* 'call' to tell the caller the extents of its data (bounding boxes
			* and times).
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool GetExtents(core::view::CallRender2D& call);

			/**
			* The render callback.
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool Render(core::view::CallRender2D& call);

			/**********************************************************************
			* variables
			**********************************************************************/

			/** The call for keyframe data */
			core::CallerSlot getDataSlot;

			/** texture resolution parameter */
			megamol::core::param::ParamSlot resolutionParam;


			// font rendering
#ifdef USE_SIMPLE_FONT
			vislib::graphics::gl::SimpleFont theFont;
#else
			vislib::graphics::gl::OutlineFont theFont;
#endif

			// the vertex buffer array for the keyframes
			vislib::Array<float> vertices;

			vislib::Array<vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> > markerTextures;

			// mouse hover
			vislib::math::Vector<float, 2> mousePos;
			int mousePosResIdx;
			bool leftMouseDown;
			bool initialClickSelection;
			// selection 
			vislib::Array<bool> selection;

			float lineLength;
			float lineYPos;

			bool updateParameters();

		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_TIMELINERENDERER_H_INCLUDED */
