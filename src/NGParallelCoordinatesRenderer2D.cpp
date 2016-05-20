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
#include "debug.h"
#include <array>
#include <iostream>
#include "vislib/graphics/gl/ShaderSource.h"

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
	//selectedItemsColor(), otherItemsColor(), axesColor(), selectionIndicatorColor(),
	dataBuffer(0), flagsBuffer(0), minimumsBuffer(0), maximumsBuffer(0),
	axisIndirectionBuffer(0), filtersBuffer(0), minmaxBuffer(0),
	itemCount(0), columnCount(0)
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
		//zen::gl::debug_message_spec{ GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, 131185 },
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

	if (!makeProgram("::pc_axes_draw::axes", this->drawAxesProgram)) return false;
	if (!makeProgram("::pc_axes_draw::scales", this->drawScalesProgram)) return false;

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

	this->drawAxesProgram.Release();
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
	auto floats = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (floats == nullptr) return;
	auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tc == nullptr) return;
	auto flagsc = getFlagsSlot.CallAs<FlagCall>();
	if (flagsc == nullptr) return;

	(*floats)(0);
	auto hash = floats->DataHash();

	if (hash == this->currentHash) return;

	this->currentHash = hash;
	(*tc)(0);
	(*flagsc)(0);

	this->computeScaling();

	this->columnCount = static_cast<GLuint>(floats->GetColumnsCount());
	this->itemCount = static_cast<GLuint>(floats->GetRowsCount());
	this->axisIndirection.resize(columnCount);
	this->filters.resize(columnCount);
	this->minimums.resize(columnCount);
	this->maximums.resize(columnCount);
	for (GLuint x = 0; x < columnCount; x++) {
		axisIndirection[x] = x;
		filters[x].dimension = 0;
		filters[x].flags = 0;
		minimums[x] = floats->GetColumnsInfos()[x].MinimumValue();
		maximums[x] = floats->GetColumnsInfos()[x].MaximumValue();
		filters[x].lower = minimums[x];
		filters[x].upper = maximums[x];
	}

	if (!flagsc->has_data()) {
		std::shared_ptr<FlagStorage::FlagVectorType> v;
		v = std::make_shared<FlagStorage::FlagVectorType>();
		v->resize(itemCount);
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
		this->enableProgramAndBind(this->drawAxesProgram);
		glUniform4fv(this->drawAxesProgram.ParameterLocation("color"), 1, this->axesColor);
		glDrawArraysInstanced(GL_LINES, 0, 2, this->columnCount);
		this->drawAxesProgram.Disable();
	}
}

void NGParallelCoordinatesRenderer2D::drawParcos(void) {

}

bool NGParallelCoordinatesRenderer2D::Render(core::view::CallRender2D& call) {
	windowAspect = static_cast<float>(call.GetViewport().AspectRatio());

	// this is the apex of suck and must die
	GLfloat modelViewMatrix_column[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
	GLfloat projMatrix_column[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
	// end suck

	this->assertData();

	auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
	if (fc == nullptr) return false;
	auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
	if (tc == nullptr) return false;

	glDisable(GL_DEPTH_TEST);

	glBegin(GL_LINES);
	for (int x = 0, max = fc->GetColumnsCount(); x < max; x++) {
		glVertex2f(this->marginX + this->axisDistance * x + 2, this->marginY);
		glVertex2f(this->marginX + this->axisDistance * x + 2, this->marginY + this->axisHeight);
	}
	glEnd();

	drawAxes();

	return true;
}
