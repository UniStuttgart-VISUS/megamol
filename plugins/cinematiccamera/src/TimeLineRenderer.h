/*
* TimeLineRenderer.h
*
*/

#ifndef MEGAMOL_CINEMATICCAMERA_TIMELINERENDERER_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_TIMELINERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/graphics/gl/Verdana.inc"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/Array.h"

#include "Keyframe.h"

// #define USE_SIMPLE_FONT

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
            // ...
            vislib::Array<vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> > markerTextures;

            vislib::math::Vector<float, 2> axisStartPos;       // joint start position for both axis

            bool                           redoAdaptation;     // indicates if the segment size/value should be recalculated

            vislib::math::Vector<float, 2> animAxisEndPos;     // end position of naimation axis
            float                          animAxisLen;        // length of the animation axis
            float                          animTotalTime;      // the total animation time
            float                          animSegmSize;       // the world space size of one segment of the animation time ruler
            float                          animSegmValue;      // the animation time value of on segment on the ruler 
            float                          animScaleFac;       // the scaling factor of the animation axis
            float                          animScaleOffset;    // (negative) offset to keep position on the ruler during scaling in focus 
            float                          animLenTimeFrac;    // the scaled fraction of the animation axis length and the total animation time
            float                          animScalePos;       // the ruler position to be kept in focus during scaling
            float                          animScaleDelta;     // animScaleOffset for new animScalePos to get new animScaleOffset for new scaling factor
            vislib::StringA                animFormatStr;      // string with adapted floating point formatting

            vislib::math::Vector<float, 2> simAxisEndPos;
            float                          simAxisLen;
            float                          simTotalTime;
            float                          simSegmSize;
            float                          simSegmValue;
            float                          simScaleFac;
            float                          simScaleOffset;
            float                          simLenTimeFrac;
            float                          simScalePos;
            float                          simScaleDelta;
            vislib::StringA                simFormatStr;

            vislib::math::Vector<float, 2> lastMousePos;
            unsigned int                   scaleAxis;

            Keyframe                       dragDropKeyframe;
            bool                           dragDropActive;
            unsigned int                   dragDropAxis;

            float                          fontSize;
            float                          keyfMarkSize;
            float                          rulerMarkSize;
            unsigned int                   fps;


            /**********************************************************************
            * functions
            **********************************************************************/

            /** Loading texture for keyframe marker. */
            bool LoadTexture(vislib::StringA filename);

            /** Draw the keyframe marker. */
            void DrawKeyframeMarker(float posX, float posY);


            /**********************************************************************
            * callback stuff
            **********************************************************************/

			/** The call for keyframe data */
            core::CallerSlot keyframeKeeperSlot;

            /**********************************************************************
            * parameter
            **********************************************************************/

            /**  */
            megamol::core::param::ParamSlot rulerFontParam;
            /**  */
            megamol::core::param::ParamSlot moveRightFrame;
            /**  */
            megamol::core::param::ParamSlot moveLeftFrame;
		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_TIMELINERENDERER_H_INCLUDED */
