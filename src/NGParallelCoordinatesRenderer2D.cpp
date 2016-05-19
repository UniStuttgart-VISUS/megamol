#include "stdafx.h"
#include "NGParallelCoordinatesRenderer2D.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/CoreInstance.h"
#include "debug.h"
#include <array>
#include <iostream>
#include "vislib/graphics/gl/ShaderSource.h"

using namespace megamol;
using namespace megamol::infovis;

NGParallelCoordinatesRenderer2D::NGParallelCoordinatesRenderer2D(void) : Renderer2DModule(),
	getDataSlot("getdata", "connects to the float table data"),
	getTFSlot("getTF", "connects to the transfer function"),
	densityFBO(),
	mousePressedX(),
	mousePressedY(),
	mouseReleasedX(),
	mouseReleasedY(),
	mouseFlags(),
	drawModeSlot("drawMode", "Draw mode"),
	drawSelectedItemsSlot("drawSelectedItems", "Draw selected items"),
	selectedItemsColorSlot("selectedItemsColor", "Color for selected items"),
	selectedItemsAlphaSlot("selectedItemsAlpha", "Alpha for selected items"),
	selectedItemsColor(),
	drawOtherItemsSlot("drawOtherItems", "Draw other (e.g., non-selected) items"),
	otherItemsColorSlot("otherItemsColor", "Color for other items (e.g., non-selected)"),
	otherItemsAlphaSlot("otherItemsAlpha", "Alpha for other items (e.g., non-selected)"),
	otherItemsColor(),
	drawAxesSlot("drawAxes", "Draw dimension axes"),
	axesColorSlot("axesColor", "Color for dimension axes"),
	axesColor(),
	selectionModeSlot("selectionMode", "Selection mode"),
	drawSelectionIndicatorSlot("drawSelectionIndicator", "Draw selection indicator"),
	selectionIndicatorColorSlot("selectionIndicatorColor", "Color for selection indicator"),
	selectionIndicatorColor(),
	pickRadiusSlot("pickRadius", "Picking radius in object-space"),
	scaleToFitSlot("scaleToFit", "fit the diagram in the viewport"),
	//scalingFactorSlot("scalingFactor", "Scaling factor"),
	//scaleFullscreenSlot("scaleFullscreen", "Scale to fullscreen"),
	//projectionMatrixSlot("projectionMatrix", "Projection matrix"),
	//viewMatrixSlot("viewMatrix", "View matrix"),
	//useCustomMatricesSlot("useCustomMatrices", "Use custom matrices"),
	//storeCamSlot("storeCam", "Store current matrices"),
	glDepthTestSlot("glEnableDepthTest", "Toggle GLDEPTHTEST"),
	glLineSmoothSlot("glEnableLineSmooth", "Toggle GLLINESMOOTH"),
	glLineWidthSlot("glLineWidth", "Value for glLineWidth"),
	resetFlagsSlot("resetFlags", "Reset item flags to initial state")
	//selectedItemsColor(), otherItemsColor(), axesColor(), selectionIndicatorColor()
{

	this->getDataSlot.SetCompatibleCall<megamol::stdplugin::datatools::floattable::CallFloatTableDataDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);

	this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
	this->MakeSlotAvailable(&this->getTFSlot);

	auto drawModes = new ::megamol::core::param::EnumParam(DRAW_DISCRETE);
	drawModes->SetTypePair(DRAW_DISCRETE, "Discrete");
	drawModes->SetTypePair(DRAW_CONTINUOUS, "Continuous");
	drawModes->SetTypePair(DRAW_HISTOGRAM, "Histogram");
	drawModeSlot.SetParameter(drawModes);
	this->MakeSlotAvailable(&drawModeSlot);
	
	drawSelectedItemsSlot << new ::megamol::core::param::BoolParam(true);
	this->MakeSlotAvailable(&drawSelectedItemsSlot);

	selectedItemsColorSlot << new ::megamol::core::param::StringParam("red");
	selectedItemsColorSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback);
	this->MakeSlotAvailable(&selectedItemsColorSlot);
	selectedItemsAlphaSlot << new param::FloatParam(1.0f, 0.0f, 1.0f);
	selectedItemsAlphaSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback);
	this->MakeSlotAvailable(&selectedItemsAlphaSlot);
	selectedItemsColorSlotCallback(selectedItemsColorSlot);

	drawOtherItemsSlot << new ::megamol::core::param::BoolParam(true);
	this->MakeSlotAvailable(&drawOtherItemsSlot);

	otherItemsColorSlot << new ::megamol::core::param::StringParam("gray");
	otherItemsColorSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::otherItemsColorSlotCallback);
	this->MakeSlotAvailable(&otherItemsColorSlot);
	otherItemsAlphaSlot << new param::FloatParam(1.0f, 0.0f, 1.0f);
	otherItemsAlphaSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::otherItemsColorSlotCallback);
	this->MakeSlotAvailable(&otherItemsAlphaSlot);
	otherItemsColorSlotCallback(otherItemsColorSlot);

	drawAxesSlot << new ::megamol::core::param::BoolParam(true);
	this->MakeSlotAvailable(&drawAxesSlot);

	axesColorSlot << new ::megamol::core::param::StringParam("indigo");
	axesColorSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::axesColorSlotCallback);
	this->MakeSlotAvailable(&axesColorSlot);
	axesColorSlotCallback(axesColorSlot);

	drawSelectionIndicatorSlot << new ::megamol::core::param::BoolParam(true);
	this->MakeSlotAvailable(&drawSelectionIndicatorSlot);

	selectionIndicatorColorSlot << new ::megamol::core::param::StringParam("MegaMolBlue");
	selectionIndicatorColorSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::selectionIndicatorColorSlotCallback);
	this->MakeSlotAvailable(&selectionIndicatorColorSlot);
	selectionIndicatorColorSlotCallback(selectionIndicatorColorSlot);

	auto pickModes = new ::megamol::core::param::EnumParam(SELECT_PICK);
	pickModes->SetTypePair(SELECT_PICK, "Pick");
	pickModes->SetTypePair(SELECT_STROKE, "Stroke");
	selectionModeSlot.SetParameter(pickModes);
	this->MakeSlotAvailable(&selectionModeSlot);

	pickRadiusSlot << new ::megamol::core::param::FloatParam(0.1f, 0.01f, 1.0f);
	this->MakeSlotAvailable(&pickRadiusSlot);
	
	//scalingFactorSlot << new ::megamol::core::param::Vector2fParam(::vislib::math::Vector< float, 2 >(1.0, 1.0));
	//this->MakeSlotAvailable(&scalingFactorSlot);
	//
	//scaleFullscreenSlot_ << new ::megamol::core::param::BoolParam(false);
	//this->MakeSlotAvailable(&scaleFullscreenSlot_);

	scaleToFitSlot << new param::BoolParam(false);
	scaleToFitSlot.SetUpdateCallback(this, &NGParallelCoordinatesRenderer2D::scalingChangedCallback);
	this->MakeSlotAvailable(&scaleToFitSlot);

	//projectionMatrixSlot_ << new ::megamol::core::param::StringParam("");
	//this->MakeSlotAvailable(&projectionMatrixSlot_);

	//viewMatrixSlot_ << new ::megamol::core::param::StringParam("");
	//this->MakeSlotAvailable(&viewMatrixSlot_);

	//useCustomMatricesSlot_ << new ::megamol::core::param::BoolParam(false);
	//this->MakeSlotAvailable(&useCustomMatricesSlot_);

	//storeCamSlot_ << new ::megamol::core::param::ButtonParam();
	//storeCamSlot_.SetUpdateCallback(this, &ParallelCoordinatesRenderer2D::storeCamSlotCallback);
	//this->MakeSlotAvailable(&storeCamSlot_);

	glDepthTestSlot << new ::megamol::core::param::BoolParam(false);
	this->MakeSlotAvailable(&glDepthTestSlot);

	glLineSmoothSlot << new ::megamol::core::param::BoolParam(false);
	this->MakeSlotAvailable(&glLineSmoothSlot);
	
	glLineWidthSlot << new ::megamol::core::param::FloatParam(1.0f, 0.1f);
	this->MakeSlotAvailable(&glLineWidthSlot);
	
	resetFlagsSlot << new ::megamol::core::param::ButtonParam();
	resetFlagsSlot.SetUpdateCallback(this, &NGParallelCoordinatesRenderer2D::resetFlagsSlotCallback);
	this->MakeSlotAvailable(&resetFlagsSlot);
}


/*
* misc::LinesRenderer::~LinesRenderer
*/
NGParallelCoordinatesRenderer2D::~NGParallelCoordinatesRenderer2D(void) {
	this->Release();
}

bool NGParallelCoordinatesRenderer2D::makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program) {
	vislib::graphics::gl::ShaderSource vert, frag;

	vislib::StringA vertname((prefix + "::vert").c_str());
	vislib::StringA fragname((prefix + "::frag").c_str());
	vislib::StringA pref(prefix.c_str());

	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(vertname, vert)) return false;
	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(fragname, frag)) return false;

	try {
		if (!program.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile %s: Unknown error\n", pref);
			return false;
		}

	} catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s (@%s): %s\n", pref,
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
			ce.FailedAction()), ce.GetMsgA());
		return false;
	} catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s: %s\n", pref, e.GetMsgA());
		return false;
	} catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s: Unknown exception\n", pref);
		return false;
	}
}

bool NGParallelCoordinatesRenderer2D::create(void) {
	std::array< zen::gl::debug_action, 2 > actions =
	{
		//zen::gl::make_debug_action_ostream(std::cerr)
		zen::gl::make_debug_action_Log(vislib::sys::Log::DefaultLog)
		, zen::gl::debug_action_throw
	};

	zen::gl::enable_debug_callback(nullptr, true, std::begin(actions), std::end(actions));
	zen::gl::enable_all_debug_messages();

	zen::gl::ignore_debug_messages(
	{
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, 1282 },
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, 131204 },
		// Buffer object ... will use VIDEO memory as the source for buffer object operations.
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, 131185 },
		// Buffer performance warning: Buffer object ... is being copied / moved from VIDEO memory to HOST memory.
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, 131186 },
	});

	if (!makeProgram("::pc_axes_draw::axes", this->drawAxesProgram)) return false;

	return true;
}

void NGParallelCoordinatesRenderer2D::release(void) {

}

bool NGParallelCoordinatesRenderer2D::MouseEvent(float x, float y, ::megamol::core::view::MouseFlags flags) {
	if (flags & ::megamol::core::view::MOUSEFLAG_MODKEY_CTRL_DOWN) {
		return false;
	}

	if (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
		if (mouseFlags != 0) {
			mouseReleasedX = x;
			mouseReleasedY = y;
		} else {
			mouseFlags = flags;
			mousePressedX = x;
			mousePressedY = y;
		}
	} else if (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_CHANGED) {
		mouseFlags = 0;
	}

	return true;
}

bool NGParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->selectedItemsColorSlot.Param<param::StringParam>()->Value(), 4, selectedItemsColor);
	selectedItemsColor[4] = static_cast<unsigned char>(this->selectedItemsAlphaSlot.Param<param::FloatParam>()->Value() * 255.0f);
	return true;
}

bool NGParallelCoordinatesRenderer2D::otherItemsColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->otherItemsColorSlot.Param<param::StringParam>()->Value(), 4, otherItemsColor);
	otherItemsColor[4] = static_cast<unsigned char>(this->otherItemsAlphaSlot.Param<param::FloatParam>()->Value() * 255.0f);
	return true;
}
bool NGParallelCoordinatesRenderer2D::axesColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->axesColorSlot.Param<param::StringParam>()->Value(), 4, axesColor);
	return true;
}
bool NGParallelCoordinatesRenderer2D::selectionIndicatorColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->selectionIndicatorColorSlot.Param<param::StringParam>()->Value(), 4, selectionIndicatorColor);
	return true;
}

bool NGParallelCoordinatesRenderer2D::scalingChangedCallback(::megamol::core::param::ParamSlot & caller) {
	this->computeScaling();
	return true;
}

bool NGParallelCoordinatesRenderer2D::resetFlagsSlotCallback(::megamol::core::param::ParamSlot & caller) {
	return true;
}

void NGParallelCoordinatesRenderer2D::assertData(void) {
	auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (fc == nullptr) return;
	auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tc == nullptr) return;

	(*fc)(1);
	auto hash = fc->DataHash();

	if (hash == this->currentHash) return;

	this->currentHash = hash;
	(*fc)(0);
	(*tc)(0);

	this->computeScaling();
}

void NGParallelCoordinatesRenderer2D::computeScaling(void) {
	auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (fc == nullptr) return;

	this->marginX = 20.f;
	this->marginY = 20.f;
	this->axisDistance = 40.0f;
	this->bounds.SetLeft(0.0f);
	this->bounds.SetRight(2.0f * marginX + this->axisDistance * (fc->GetColumnsCount() - 1));

	if (this->scaleToFitSlot.Param<param::BoolParam>()->Value()) {
		// scale to fit
		float requiredHeight = this->bounds.Width() / windowAspect;
		this->axisHeight = requiredHeight - 2.0f * marginY;
	} else {
		this->axisHeight = 80.0f;
	}
	this->bounds.SetBottom(0.0f);
	this->bounds.SetTop(2.0f * marginY + this->axisHeight);
}

bool NGParallelCoordinatesRenderer2D::GetExtents(core::view::CallRender2D& call) {
	windowAspect = call.GetViewport().AspectRatio();

	this->assertData();
	
	call.SetBoundingBox(this->bounds);

	return true;
}

bool NGParallelCoordinatesRenderer2D::Render(core::view::CallRender2D& call) {
	windowAspect = call.GetViewport().AspectRatio();

	this->assertData();

	auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (fc == nullptr) return false;
	auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tc == nullptr) return false;

	glBegin(GL_LINES);
	for (int x = 0, max = fc->GetColumnsCount(); x < max; x++) {
		glVertex2f(this->marginX + this->axisDistance * x, this->marginY);
		glVertex2f(this->marginX + this->axisDistance * x, this->marginY + this->axisHeight);
	}
	glEnd();

	return true;
}
