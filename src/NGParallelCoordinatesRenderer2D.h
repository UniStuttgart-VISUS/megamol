#ifndef MEGAMOL_INFOVIS_NGPARCORENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_NGPARCORENDERER2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

namespace megamol {
namespace infovis {

	using namespace megamol::core;

	class NGParallelCoordinatesRenderer2D : public view::Renderer2DModule {
	public:

		enum // draw mode
		{
			DRAW_DISCRETE = 0
			, DRAW_CONTINUOUS
			, DRAW_HISTOGRAM
		};

		enum // selection mode
		{
			SELECT_PICK = 0
			, SELECT_STROKE
		};

		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static inline const char *ClassName(void) {
			return "NGParallelCoordinatesRenderer2D";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static inline const char *Description(void) {
			return "Parallel Coordinates Renderer for generic Float Tables";
		}

		/**
		* Answers whether this module is available on the current system.
		*
		* @return 'true' if the module is available, 'false' otherwise.
		*/
		static inline bool IsAvailable(void) {
			// TODO unknown yet
			return true;
		}

		/**
		* Initialises a new instance.
		*/
		NGParallelCoordinatesRenderer2D(void);

		/**
		* Finalises an instance.
		*/
		virtual ~NGParallelCoordinatesRenderer2D(void);

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
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		virtual bool Render(core::view::CallRender2D& call);

		virtual bool GetExtents(core::view::CallRender2D& call);

		virtual bool MouseEvent(float x, float y, ::megamol::core::view::MouseFlags flags);

		bool selectedItemsColorSlotCallback(::megamol::core::param::ParamSlot & caller);
		bool otherItemsColorSlotCallback(::megamol::core::param::ParamSlot & caller);
		bool axesColorSlotCallback(::megamol::core::param::ParamSlot & caller);
		bool selectionIndicatorColorSlotCallback(::megamol::core::param::ParamSlot & caller);
		bool scalingChangedCallback(::megamol::core::param::ParamSlot & caller);
		bool resetFlagsSlotCallback(::megamol::core::param::ParamSlot & caller);

	private:

		void assertData(void);

		void computeScaling(void);

		void drawAxes(void);

		void drawParcos(void);

		bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program);

		CallerSlot getDataSlot;

		CallerSlot getTFSlot;

		size_t currentHash;

		::vislib::graphics::gl::FramebufferObject densityFBO;

		float mousePressedX;
		float mousePressedY;
		float mouseReleasedX;
		float mouseReleasedY;
		::megamol::core::view::MouseFlags mouseFlags;

		::megamol::core::param::ParamSlot drawModeSlot;

		::megamol::core::param::ParamSlot drawSelectedItemsSlot;
		::megamol::core::param::ParamSlot selectedItemsColorSlot;
		::megamol::core::param::ParamSlot selectedItemsAlphaSlot;
		unsigned char selectedItemsColor[4];

		::megamol::core::param::ParamSlot drawOtherItemsSlot;
		::megamol::core::param::ParamSlot otherItemsColorSlot;
		::megamol::core::param::ParamSlot otherItemsAlphaSlot;
		unsigned char otherItemsColor[4];

		::megamol::core::param::ParamSlot drawAxesSlot;
		::megamol::core::param::ParamSlot axesColorSlot;
		unsigned char axesColor[4];

		::megamol::core::param::ParamSlot selectionModeSlot;
		::megamol::core::param::ParamSlot drawSelectionIndicatorSlot;
		::megamol::core::param::ParamSlot selectionIndicatorColorSlot;
		unsigned char selectionIndicatorColor[4];

		::megamol::core::param::ParamSlot pickRadiusSlot;

		::megamol::core::param::ParamSlot scaleToFitSlot;
		//::megamol::core::param::ParamSlot scalingFactorSlot;
		//::megamol::core::param::ParamSlot scaleFullscreenSlot;
		//::megamol::core::param::ParamSlot projectionMatrixSlot;
		//::megamol::core::param::ParamSlot viewMatrixSlot;
		//::megamol::core::param::ParamSlot useCustomMatricesSlot;

		//::megamol::core::param::ParamSlot storeCamSlot;
		//bool storeCamSlotCallback(::megamol::core::param::ParamSlot & caller);

		::megamol::core::param::ParamSlot glDepthTestSlot;
		::megamol::core::param::ParamSlot glLineSmoothSlot;
		::megamol::core::param::ParamSlot glLineWidthSlot;

		::megamol::core::param::ParamSlot resetFlagsSlot;

		float marginX, marginY;
		float axisDistance;
		float axisHeight;
		float fontSize;
		float windowAspect;
		vislib::math::Rectangle<float> bounds;

		vislib::graphics::gl::GLSLShader drawAxesProgram;
	};

} /* end namespace infovis */
} /* end namespace megamol */

#endif /* MEGAMOL_INFOVIS_NGPARCORENDERER2D_H_INCLUDED */