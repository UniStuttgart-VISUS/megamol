#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "NGParallelCoordinatesRenderer2D.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "FlagCall.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/CoreInstance.h"
#include <array>
#include <iostream>
#include "vislib/graphics/gl/ShaderSource.h"
#include "debug.h"

//#define FUCK_THE_PIPELINE
//#define USE_TESSELLATION
//#define BE_DEBUGGABLE

using namespace megamol;
using namespace megamol::infovis;

NGParallelCoordinatesRenderer2D::NGParallelCoordinatesRenderer2D(void) : Renderer2DModule(),
	getDataSlot("getdata", "connects to the float table data"),
	getTFSlot("getTF", "connects to the transfer function"),
	getFlagsSlot("getFlags", "connects to the flag storage"),
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
	filterIndicatorColorSlot("filterIndicatorCol", "Color for filter indicators"),
	filterIndicatorColor(),
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
	resetFlagsSlot("resetFlags", "Reset item flags to initial state"),
	resetFiltersSlot("resetFilters", "Reset dimension filters to initial state"),
	//selectedItemsColor(), otherItemsColor(), axesColor(), selectionIndicatorColor(),
	dataBuffer(0), flagsBuffer(0), minimumsBuffer(0), maximumsBuffer(0),
	axisIndirectionBuffer(0), filtersBuffer(0), minmaxBuffer(0),
	itemCount(0), columnCount(0), dragging(false), filtering(false),
	numTicks(5), pickedAxis(-1), pickedIndicatorAxis(-1), pickedIndicatorIndex(-1)
{

	this->getDataSlot.SetCompatibleCall<megamol::stdplugin::datatools::floattable::CallFloatTableDataDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);

	this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
	this->MakeSlotAvailable(&this->getTFSlot);

	this->getFlagsSlot.SetCompatibleCall<FlagCallDescription>();
	this->MakeSlotAvailable(&this->getFlagsSlot);

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

	axesColorSlot << new ::megamol::core::param::StringParam("white");
	axesColorSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::axesColorSlotCallback);
	this->MakeSlotAvailable(&axesColorSlot);
	axesColorSlotCallback(axesColorSlot);

	filterIndicatorColorSlot << new ::megamol::core::param::StringParam("orange");
	filterIndicatorColorSlot.SetUpdateCallback(&NGParallelCoordinatesRenderer2D::filterIndicatorColorSlotCallback);
	this->MakeSlotAvailable(&filterIndicatorColorSlot);
	filterIndicatorColorSlotCallback(filterIndicatorColorSlot);

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

	resetFiltersSlot << new ::megamol::core::param::ButtonParam();
	resetFiltersSlot.SetUpdateCallback(this, &NGParallelCoordinatesRenderer2D::resetFiltersSlotCallback);
	this->MakeSlotAvailable(&resetFiltersSlot);

	fragmentMinMax.resize(2);
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
	return true;
}

bool NGParallelCoordinatesRenderer2D::makeComputeProgram(std::string prefix, vislib::graphics::gl::GLSLComputeShader& program) {
	vislib::graphics::gl::ShaderSource comp;

	vislib::StringA compname((prefix + "::comp").c_str());
	vislib::StringA pref(prefix.c_str());

	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(compname, comp)) return false;

	try {
		if (!program.Compile(comp.Code(), comp.Count())) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile %s: Unknown error\n", pref);
			return false;
		}
		if (!program.Link()) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to link %s: Unknown error\n", pref);
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
	return true;
}

bool NGParallelCoordinatesRenderer2D::makeTessellationProgram(std::string prefix, vislib::graphics::gl::GLSLTesselationShader& program) {
	vislib::graphics::gl::ShaderSource vert, frag, control, eval, geom;

	vislib::StringA vertname((prefix + "::vert").c_str());
	vislib::StringA fragname((prefix + "::frag").c_str());
	vislib::StringA controlname((prefix + "::control").c_str());
	vislib::StringA evalname((prefix + "::eval").c_str());
	vislib::StringA geomname((prefix + "::geom").c_str());
	vislib::StringA pref(prefix.c_str());

	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(vertname, vert)) return false;
	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(fragname, frag)) return false;
	// no complete tess?
	auto r1 = this->instance()->ShaderSourceFactory().MakeShaderSource(controlname, control);
	auto r2 = this->instance()->ShaderSourceFactory().MakeShaderSource(evalname, eval);
	if (r1 != r2) return false;
	bool haveTess = r1;
	bool haveGeom = this->instance()->ShaderSourceFactory().MakeShaderSource(geomname, geom);
	
	try {
		if (!program.Compile(vert.Code(), vert.Count(),
			haveTess ? control.Code() : nullptr, haveTess ? control.Count() : 0,
			haveTess ? eval.Code() : nullptr, haveTess ? eval.Count() : 0,
			haveGeom ? geom.Code() : nullptr, haveGeom ? geom.Count() : 0,
			frag.Code(), frag.Count())) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile %s: Unknown error\n", pref);
			return false;
		}
		if (!program.Link()) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to link %s: Unknown error\n", pref);
			return false;
		}

	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s (@%s): %s\n", pref,
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
			ce.FailedAction()), ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s: %s\n", pref, e.GetMsgA());
		return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s: Unknown exception\n", pref);
		return false;
	}
	return true;
}

bool NGParallelCoordinatesRenderer2D::enableProgramAndBind(vislib::graphics::gl::GLSLShader& program) {
	program.Enable();
	// bindbuffer?
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dataBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, flagsBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, minimumsBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, maximumsBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, axisIndirectionBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, filtersBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, minmaxBuffer);

	glUniform2f(program.ParameterLocation("scaling"), 1.0f, 1.0f); // scaling, whatever
	glUniformMatrix4fv(program.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
	glUniformMatrix4fv(program.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);
	glUniform1ui(program.ParameterLocation("dimensionCount"), this->columnCount);
	glUniform1ui(program.ParameterLocation("itemCount"), this->itemCount);

	glUniform2f(program.ParameterLocation("margin"), this->marginX, this->marginY);
	glUniform1f(program.ParameterLocation("axisDistance"), this->axisDistance);
	glUniform1f(program.ParameterLocation("axisHeight"), this->axisHeight);

	return true;
}

bool NGParallelCoordinatesRenderer2D::create(void) {
	std::array< zen::gl::debug_action, 1 > actions =
	{
		//zen::gl::make_debug_action_ostream(std::cerr)
		zen::gl::make_debug_action_Log(vislib::sys::Log::DefaultLog)
		//, zen::gl::debug_action_throw
	};

	zen::gl::enable_debug_callback(nullptr, true, std::begin(actions), std::end(actions));
	zen::gl::enable_all_debug_messages();

	zen::gl::ignore_debug_messages(
	{
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, 1282 },
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, 131204 },
		// Buffer object ... will use VIDEO memory as the source for buffer object operations.
		zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, 131185 },
		zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, 131188 },
		zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, 131184 }
		// Buffer performance warning: Buffer object ... is being copied / moved from VIDEO memory to HOST memory.
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, 131186 },
	});

	glGenBuffers(1, &dataBuffer);
	glGenBuffers(1, &flagsBuffer);
	glGenBuffers(1, &minimumsBuffer);
	glGenBuffers(1, &maximumsBuffer);
	glGenBuffers(1, &axisIndirectionBuffer);
	glGenBuffers(1, &filtersBuffer);
	glGenBuffers(1, &minmaxBuffer);
	glGenBuffers(1, &counterBuffer);

#ifndef BE_DEBUGGABLE
	if (!font.Initialise()) return false;
#endif

	if (!makeProgram("::pc_axes_draw::axes", this->drawAxesProgram)) return false;
	if (!makeProgram("::pc_axes_draw::scales", this->drawScalesProgram)) return false;
	if (!makeProgram("::pc_axes_draw::filterindicators", this->drawFilterIndicatorsProgram)) return false;

	if (!makeProgram("::pc_item_draw::discrete", this->drawItemsDiscreteProgram)) return false;
	if (!makeProgram("::pc_item_draw::muhaha", this->traceItemsDiscreteProgram)) return false;

	if (!makeTessellationProgram("::pc_item_draw::discTess", drawItemsDiscreteTessProgram)) return false;
	glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &this->maxAxes); // TODO we should reject data with more axes!
	this->isoLinesPerInvocation = maxAxes; // warning: for tesslevel n there are JUST n lines!!! not n+1 !!

	if (!makeProgram("::fragment_count", this->drawItemContinuousProgram)) return false;
	if (!makeComputeProgram("::fragment_count", this->minMaxProgram)) return false;

	//if (!makeProgram("::pc_item_draw::histogram", this->drawItemsHistogramProgram)) return false;

	if (!makeComputeProgram("::pc_item_filter", this->filterProgram)) return false;

	glGetProgramiv(this->filterProgram, GL_COMPUTE_LOCAL_WORK_SIZE, filterWorkgroupSize);
	glGetProgramiv(this->minMaxProgram, GL_COMPUTE_LOCAL_WORK_SIZE, counterWorkgroupSize);

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount[0]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount[1]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount[2]);

	return true;
}

void NGParallelCoordinatesRenderer2D::release(void) {
	glDeleteBuffers(1, &dataBuffer);
	glDeleteBuffers(1, &flagsBuffer);
	glDeleteBuffers(1, &minimumsBuffer);
	glDeleteBuffers(1, &maximumsBuffer);
	glDeleteBuffers(1, &axisIndirectionBuffer);
	glDeleteBuffers(1, &filtersBuffer);
	glDeleteBuffers(1, &minmaxBuffer);
	glDeleteBuffers(1, &counterBuffer);

	this->drawAxesProgram.Release();
}

int NGParallelCoordinatesRenderer2D::mouseXtoAxis(float x) {
	float f = (x - this->marginX) / this->axisDistance;
	float frac = f - static_cast<long>(f);
	int integral = static_cast<int>(std::round(f));
	if (integral >= this->columnCount || integral < 0) return -1;
	if (frac > 0.8 || frac < 0.2) {
		//vislib::sys::Log::DefaultLog.WriteInfo("picking axis %i at mouse position of axis %i", axisIndirection[integral], integral);
		return axisIndirection[integral];
	} else {
		return -1;
	}
}

void NGParallelCoordinatesRenderer2D::pickIndicator(float x, float y, int& axis, int& index) {
	axis = mouseXtoAxis(x);
	float val = (y - this->marginY) / axisHeight;
	if (val >= 0.0f && val <= 1.0f && axis != -1) {
		//float thresh = this->maximums[axis] - this->minimums[axis];
		//thresh /= 10.0f;
		float thresh = 0.1f;
		//val = relToAbsValue(axis, val);
		if (fabs(this->filters[axis].upper - val) < thresh) {
			index = 1;
		} else if (fabs(this->filters[axis].lower - val) < thresh) {
			index = 0;
		} else {
			index = -1;
		}
	}
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
	if (pickedAxis != -1 && (fabs(mousePressedX - x) > this->axisDistance * 0.5f)
		&& (flags & ::megamol::core::view::MOUSEFLAG_MODKEY_ALT_DOWN)
		&& (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN)) {
		this->dragging = true;
	} else {
		this->dragging = false;
	}
	if ((flags & ::megamol::core::view::MOUSEFLAG_MODKEY_ALT_DOWN)
		&& (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_CHANGED)
		&& !(flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN)
		&& !dragging) {
		pickedAxis = mouseXtoAxis(mouseReleasedX);
	}

	if ((flags & ::megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_DOWN)
		&& (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_CHANGED)
		&& !(flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN)
		&& !dragging) {
		pickIndicator(mouseReleasedX, mouseReleasedY, pickedIndicatorAxis, pickedIndicatorIndex);
	}
	if ((pickedIndicatorAxis != -1)
		&& (flags & ::megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_DOWN)
		&& (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN)
		&& !dragging) {
		this->filtering = true;
	} else {
		this->filtering = false;
	}
	if ((flags & ::megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_DOWN)
		&& (flags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN)
		&& filtering) {
		int checkAxis, checkIndex;
		pickIndicator(mouseX, mouseY, checkAxis, checkIndex);
		if (pickedIndicatorAxis != -1 && checkAxis == pickedIndicatorAxis && checkIndex == pickedIndicatorIndex) {
			float val = (mouseReleasedY - this->marginY) / axisHeight;
			val = (std::max)(0.0f, val);
			val = (std::min)(val, 1.0f);
			//if (val >= 0.0f && val <= 1.0f) {
				//val = relToAbsValue(pickedIndicatorAxis, val);
				if (pickedIndicatorIndex == 0) {
					this->filters[pickedIndicatorAxis].lower = val;
				} else {
					this->filters[pickedIndicatorAxis].upper = val;
				}
			//}
		} else {
			filtering = false;
		}
	}

	mouseX = x;
	mouseY = y;

	return true;
}

bool NGParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->selectedItemsColorSlot.Param<param::StringParam>()->Value(), 4, selectedItemsColor);
	selectedItemsColor[3] = this->selectedItemsAlphaSlot.Param<param::FloatParam>()->Value();
	return true;
}

bool NGParallelCoordinatesRenderer2D::otherItemsColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->otherItemsColorSlot.Param<param::StringParam>()->Value(), 4, otherItemsColor);
	otherItemsColor[3] = this->otherItemsAlphaSlot.Param<param::FloatParam>()->Value();
	return true;
}
bool NGParallelCoordinatesRenderer2D::axesColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->axesColorSlot.Param<param::StringParam>()->Value(), 4, axesColor);
	return true;
}
bool NGParallelCoordinatesRenderer2D::filterIndicatorColorSlotCallback(::megamol::core::param::ParamSlot & caller) {
	utility::ColourParser::FromString(this->filterIndicatorColorSlot.Param<param::StringParam>()->Value(), 4, filterIndicatorColor);
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

bool NGParallelCoordinatesRenderer2D::resetFiltersSlotCallback(::megamol::core::param::ParamSlot & caller) {
	for (auto i = 0; i < this->columnCount; i++) {
		this->filters[i].lower = 0.0f;
		this->filters[i].upper = 1.0f;
	}
	return true;
}

void NGParallelCoordinatesRenderer2D::assertData(void) {
	auto floats = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (floats == nullptr) return;
	auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tc == nullptr) return;
	auto flagsc = getFlagsSlot.CallAs<FlagCall>();
	if (flagsc == nullptr) return;

	(*floats)(0);
	auto hash = floats->DataHash();
	(*tc)(0);
	(*flagsc)(0);

	if (hash == this->currentHash) return;

	this->currentHash = hash;

	this->computeScaling();

	this->columnCount = static_cast<GLuint>(floats->GetColumnsCount());
	this->itemCount = static_cast<GLuint>(floats->GetRowsCount());
	this->axisIndirection.resize(columnCount);
	this->filters.resize(columnCount);
	this->minimums.resize(columnCount);
	this->maximums.resize(columnCount);
	this->names.resize(columnCount);
	for (GLuint x = 0; x < columnCount; x++) {
		axisIndirection[x] = x;
		filters[x].dimension = 0;
		filters[x].flags = 0;
		minimums[x] = floats->GetColumnsInfos()[x].MinimumValue();
		maximums[x] = floats->GetColumnsInfos()[x].MaximumValue();
		names[x] = floats->GetColumnsInfos()[x].Name();
		filters[x].lower = 0.0f; // minimums[x];
		filters[x].upper = 1.0f; // maximums[x];
	}

	if (!flagsc->has_data() || flagsc->GetFlags().size() != itemCount) {
		std::shared_ptr<FlagStorage::FlagVectorType> v;
		v = std::make_shared<FlagStorage::FlagVectorType>();
		v->assign(itemCount, FlagStorage::ENABLED);
		flagsc->SetFlags(v);
		(*flagsc)(1); // set flags
	}

	auto flagvector = flagsc->GetFlags();

	//dataBuffer, flagsBuffer, minimumsBuffer, maximumsBuffer, axisIndirectionBuffer, filtersBuffer, minmaxBuffer;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, dataBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * this->itemCount * sizeof(float), floats->GetData(), GL_STATIC_DRAW); // TODO: huh.
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, flagsBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->itemCount * sizeof(FlagStorage::FlagItemType), flagvector.data(), GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, minimumsBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(GLfloat), this->minimums.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, maximumsBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(GLfloat), this->maximums.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, axisIndirectionBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(GLuint), axisIndirection.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, filtersBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(DimensionFilter), this->filters.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, minmaxBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLfloat), fragmentMinMax.data(), GL_DYNAMIC_READ); // TODO: huh.
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
		this->axisHeight = requiredHeight - 3.0f * marginY;
	} else {
		this->axisHeight = 80.0f;
	}
	this->bounds.SetBottom(0.0f);
	this->bounds.SetTop(3.0f * marginY + this->axisHeight);
}

bool NGParallelCoordinatesRenderer2D::GetExtents(core::view::CallRender2D& call) {
	windowAspect = static_cast<float>(call.GetViewport().AspectRatio());

	this->assertData();
	
	call.SetBoundingBox(this->bounds);

	return true;
}

void NGParallelCoordinatesRenderer2D::drawAxes(void) {
	if (this->columnCount > 0) {

		//if ((mouseFlags & ::megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN)
		//	&& (mouseFlags & ::megamol::core::view::MOUSEFLAG_MODKEY_ALT_DOWN)
		//	&& pickedAxis != -1) {
		if (dragging) {
			// we are dragging an axis!

			int currAxis = mouseXtoAxis(mouseX);
			//printf("trying to drag to axis %i\n", currAxis);
			if (currAxis != pickedAxis && currAxis >= 0 && currAxis < this->columnCount) {
				for (auto ax = this->axisIndirection.begin(), e = this->axisIndirection.end(); ax != e; ax++) {
					if (*ax == pickedAxis) {
						this->axisIndirection.erase(ax);
						break;
					}
				}
				for (auto ax = this->axisIndirection.begin(), e = this->axisIndirection.end(); ax != e; ax++) {
					if (*ax == currAxis) {
						if (mouseX > mousePressedX) {
							ax++;
							this->axisIndirection.insert(ax, pickedAxis);
						} else {
							this->axisIndirection.insert(ax, pickedAxis);
						}
						break;
					}
				}
			}

		}

		this->enableProgramAndBind(this->drawAxesProgram);
		glUniform4fv(this->drawAxesProgram.ParameterLocation("color"), 1, this->axesColor);
		glUniform1i(this->drawAxesProgram.ParameterLocation("pickedAxis"), pickedAxis);
		glDrawArraysInstanced(GL_LINES, 0, 2, this->columnCount);
		this->drawAxesProgram.Disable();

		this->enableProgramAndBind(this->drawScalesProgram);
		glUniform4fv(this->drawScalesProgram.ParameterLocation("color"), 1, this->axesColor);
		glUniform1ui(this->drawScalesProgram.ParameterLocation("numTicks"), this->numTicks);
		glUniform1f(this->drawScalesProgram.ParameterLocation("axisHalfTick"), 2.0f);
		glUniform1i(this->drawScalesProgram.ParameterLocation("pickedAxis"), pickedAxis);
		glDrawArraysInstanced(GL_LINES, 0, 2, this->columnCount * this->numTicks);
		this->drawScalesProgram.Disable();

		this->enableProgramAndBind(this->drawFilterIndicatorsProgram);
		glUniform4fv(this->drawFilterIndicatorsProgram.ParameterLocation("color"), 1, this->filterIndicatorColor);
		glUniform1f(this->drawFilterIndicatorsProgram.ParameterLocation("axisHalfTick"), 2.0f);
		glUniform2i(this->drawFilterIndicatorsProgram.ParameterLocation("pickedIndicator"), pickedIndicatorAxis, pickedIndicatorIndex);
		glDrawArraysInstanced(GL_LINE_STRIP, 0, 3, this->columnCount * 2);
		this->drawScalesProgram.Disable();

#ifndef BE_DEBUGGABLE
		glActiveTexture(GL_TEXTURE0);
		for (unsigned int c = 0; c < this->columnCount; c++) {
			unsigned int realCol = this->axisIndirection[c];
			if (this->pickedAxis == realCol) {
				glColor3f(1.0f, 0.0f, 0.0f);
			} else {
				glColor3fv(this->axesColor);
			}
			float x = this->marginX + this->axisDistance * c;
			float fontsize = this->axisDistance / 10.0f;
			this->font.DrawString(x, this->marginY * 0.5f                   , fontsize, true, std::to_string(minimums[realCol]).c_str(), vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
			this->font.DrawString(x, this->marginY * 1.5f + this->axisHeight, fontsize, true, std::to_string(maximums[realCol]).c_str(), vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
			this->font.DrawString(x, this->marginY * 2.5f + this->axisHeight, fontsize*2.0f, true, names[realCol].c_str(), vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
		}
#endif

	}
}

void NGParallelCoordinatesRenderer2D::drawDiscrete(const float otherColor[4], const float selectedColor[4], float tfColorFactor) {
	if (this->drawOtherItemsSlot.Param<param::BoolParam>()->Value()) {
		this->drawItemsDiscrete(FlagStorage::ENABLED | FlagStorage::SELECTED | FlagStorage::FILTERED, FlagStorage::ENABLED, otherColor, tfColorFactor);
	}
	//if (this->drawSelectedItemsSlot.Param<param::BoolParam>()->Value()) {
	//	this->drawItemsDiscrete(FlagStorage::ENABLED | FlagStorage::SELECTED | FlagStorage::FILTERED, FlagStorage::ENABLED | FlagStorage::SELECTED, selectedColor, tfColorFactor);
	//}
}

void NGParallelCoordinatesRenderer2D::drawItemsDiscrete(uint32_t testMask, uint32_t passMask, const float color[4], float tfColorFactor) {
	auto tf = this->getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tf == nullptr) return;

#ifdef FUCK_THE_PIPELINE
	vislib::graphics::gl::GLSLShader& prog = this->traceItemsDiscreteProgram;
#else
#ifdef USE_TESSELLATION
	vislib::graphics::gl::GLSLShader& prog = this->drawItemsDiscreteTessProgram;
#else
	vislib::graphics::gl::GLSLShader& prog = this->drawItemsDiscreteProgram;
#endif
#endif

	this->enableProgramAndBind(prog);
	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_1D, tf->OpenGLTexture());
	glUniform4fv(prog.ParameterLocation("color"), 1, color);
	glUniform1f(prog.ParameterLocation("tfColorFactor"), tfColorFactor);
	glUniform1i(prog.ParameterLocation("transferFunction"), 5);
	glUniform1ui(prog.ParameterLocation("fragmentTestMask"), testMask);
	glUniform1ui(prog.ParameterLocation("fragmentPassMask"), passMask);

#ifdef FUCK_THE_PIPELINE
	glDrawArrays(GL_TRIANGLES, 0, 6 * ((this->itemCount / 128) + 1));
#else
#ifdef USE_TESSELLATION
	glUniform1i(prog.ParameterLocation("isoLinesPerInvocation"), isoLinesPerInvocation);
	glPatchParameteri(GL_PATCH_VERTICES, 1);
	glDrawArrays(GL_PATCHES, 0, (this->itemCount / isoLinesPerInvocation) + 1);
#else
	//glDrawArraysInstanced(GL_LINE_STRIP, 0, this->columnCount, this->itemCount);
	//glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, this->columnCount * 2, this->itemCount);
	glDrawArrays(GL_LINES, 0, (this->columnCount - 1) * 2 * this->itemCount);
	//glDrawArrays(GL_TRIANGLES, 0, (this->columnCount - 1) * 6 * this->itemCount);
#endif
#endif
	prog.Disable();
}

void NGParallelCoordinatesRenderer2D::doFragmentCount(void) {
	int invocations[] = {
		static_cast<int>(std::ceil(windowWidth / 16)),
		static_cast<int>(std::ceil(windowHeight / 16))
	};
	GLuint invocationCount = invocations[0] * invocations[1];

	size_t bytes = sizeof(uint32_t) * 2 * invocationCount;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_STATIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, counterBuffer);

	glActiveTexture(GL_TEXTURE1);
	densityFBO.BindColourTexture();

	GLuint groupCounts[3] = {
		static_cast<GLuint>((std::max)(1.0f, std::ceil(float(invocations[0]) / counterWorkgroupSize[0]))),
		static_cast<GLuint>((std::max)(1.0f, std::ceil(float(invocations[1]) / counterWorkgroupSize[1]))),
		1
	};

	this->enableProgramAndBind(minMaxProgram);

	// uniforms invocationcount etc.
	::glUniform1ui(minMaxProgram.ParameterLocation("invocationCount"), invocationCount);
	::glUniform4fv(minMaxProgram.ParameterLocation("clearColor"), 1, backgroundColor);

	minMaxProgram.Dispatch(groupCounts[0], groupCounts[1], groupCounts[2]);

	minMaxProgram.Disable();
}

void NGParallelCoordinatesRenderer2D::drawItemsContinuous(void) {
	doFragmentCount();
	this->enableProgramAndBind(drawItemContinuousProgram);
	//glUniform2f(drawItemContinuousProgram.ParameterLocation("bottomLeft"), 0.0f, 0.0f);
	//glUniform2f(drawItemContinuousProgram.ParameterLocation("topRight"), windowWidth, windowHeight);
	glActiveTexture(GL_TEXTURE1);
	densityFBO.BindColourTexture();
	glUniform4fv(this->drawItemContinuousProgram.ParameterLocation("clearColor"), 1, backgroundColor);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	drawItemContinuousProgram.Disable();
}

void NGParallelCoordinatesRenderer2D::drawItemsHistogram(void) {
	doFragmentCount();
	this->enableProgramAndBind(drawItemsHistogramProgram);
	glActiveTexture(GL_TEXTURE1);
	densityFBO.BindColourTexture();
	glUniform4fv(this->drawItemContinuousProgram.ParameterLocation("clearColor"), 1, backgroundColor);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	drawItemContinuousProgram.Disable();
}

void NGParallelCoordinatesRenderer2D::drawParcos(void) {

	// TODO only when filters changed!
	size_t groups = this->itemCount / (filterWorkgroupSize[0] * filterWorkgroupSize[1] * filterWorkgroupSize[2]);
	GLuint groupCounts[3] = {
		(groups % maxWorkgroupCount[0]) + 1u
		, (groups / maxWorkgroupCount[0]) + 1u
		, 1u
	};
	this->enableProgramAndBind(this->filterProgram);
	filterProgram.Dispatch(groupCounts[0], groupCounts[1], groupCounts[2]);

	const float red[] = { 1.0f, 0.0f, 0.0f, 1.0 };
	const float moreRed[] = { 10.0f, 0.0f, 0.0f, 1.0 };

	auto drawmode = this->drawModeSlot.Param<param::EnumParam>()->Value();

	switch (drawmode) {
		case DRAW_DISCRETE:
			this->drawDiscrete(this->otherItemsColor, this->selectedItemsColor, 1.0f);
			break;
		case DRAW_CONTINUOUS:
		case DRAW_HISTOGRAM:
			bool ok = true;
			if (!this->densityFBO.IsValid() ||
				this->densityFBO.GetWidth() != windowWidth || this->densityFBO.GetHeight() != windowHeight) {
				densityFBO.Release();
				ok = densityFBO.Create(windowWidth, windowHeight, GL_R32F, GL_RED, GL_FLOAT);
			}
			if (ok) {
				densityFBO.Enable();
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				//::glDisable(GL_ALPHA_TEST);
				glDisable(GL_DEPTH_TEST);
				glEnable(GL_BLEND);
				glBlendFunc(GL_ONE, GL_ONE);
				glBlendEquation(GL_FUNC_ADD);
				this->drawDiscrete(red, moreRed, 0.0f);
				densityFBO.Disable();
				glDisable(GL_BLEND);

				if (drawmode == DRAW_CONTINUOUS) {
					this->drawItemsContinuous();
				} else if (drawmode == DRAW_HISTOGRAM) {
					this->drawItemsHistogram();
				}

			} else {
				vislib::sys::Log::DefaultLog.WriteError("could not create FBO");
			}
			break;
	}


}

bool NGParallelCoordinatesRenderer2D::Render(core::view::CallRender2D& call) {
	windowAspect = static_cast<float>(call.GetViewport().AspectRatio());

	// this is the apex of suck and must die
	glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
	glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
	// end suck
	windowWidth = call.GetViewport().Width();
	windowHeight = call.GetViewport().Height();
	auto bg = call.GetBackgroundColour();
	backgroundColor[0] = bg[0] / 255.0f;
	backgroundColor[1] = bg[1] / 255.0f;
	backgroundColor[2] = bg[2] / 255.0f;
	backgroundColor[3] = bg[3] / 255.0f;

	this->assertData();

	auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (fc == nullptr) return false;
	auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tc == nullptr) return false;

	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, axisIndirectionBuffer);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->columnCount * sizeof(GLuint), axisIndirection.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, filtersBuffer);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->columnCount * sizeof(DimensionFilter), this->filters.data());

	drawParcos();

	drawAxes();

	glDepthMask(GL_TRUE);

	return true;
}
