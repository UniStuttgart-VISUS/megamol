/*
* TimeLineRenderer.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_TIMELINERENDERER_H_INCLUDED
#define MEGAMOL_CINEMATIC_TIMELINERENDERER_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"

#include "vislib/graphics/Cursor2D.h"
#include "vislib/graphics/InputModifiers.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"

#include "Keyframe.h"
#include "CallKeyframeKeeper.h"
#include "CinematicUtils.h"


namespace megamol {
namespace cinematic {

	/**
	* Timeline rendering.
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
			return "Renders the timeline of keyframes";
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

        /** 
        * The mouse button pressed/released callback. 
        */
        virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) override;

        /** 
        * The mouse movement callback.
        */
        virtual bool OnMouseMove(double x, double y) override;

	private:
			
        /**********************************************************************
        * variables
        **********************************************************************/

        CinematicUtils utils;
        megamol::core::utility::SDFFont theFont;
        GLuint texture;

        glm::vec2                      axisStartPos;       // joint start position for both axis

        glm::vec2                      animAxisEndPos;     // end position of animation axis
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

        glm::vec2                      simAxisEndPos;
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

        unsigned int                   scaleAxis;

        Keyframe                       dragDropKeyframe;
        bool                           dragDropActive;
        unsigned int                   dragDropAxis;

        float                          fontSize;
        float                          keyfMarkSize;
        float                          rulerMarkSize;
        unsigned int                   fps;
        glm::vec2                      viewport;

        // INPUT --------------------------------------------------------------

        float mouseX;
        float mouseY;
        float  lastMouseX;
        float lastMouseY;
        core::view::MouseButton mouseButton;
        core::view::MouseButtonAction mouseAction;

        /**********************************************************************
        * functions
        **********************************************************************/

        void pushMarkerTexture(float pos_x, float pos_y, glm::vec4 color);

        void adaptAxis(void);

        /**********************************************************************
        * callback stuff
        **********************************************************************/

        core::CallerSlot keyframeKeeperSlot;

        /**********************************************************************
        * parameter
        **********************************************************************/

        megamol::core::param::ParamSlot rulerFontParam;
        megamol::core::param::ParamSlot moveRightFrameParam;
        megamol::core::param::ParamSlot moveLeftFrameParam;
        megamol::core::param::ParamSlot resetPanScaleParam;
	};

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_TIMELINERENDERER_H_INCLUDED