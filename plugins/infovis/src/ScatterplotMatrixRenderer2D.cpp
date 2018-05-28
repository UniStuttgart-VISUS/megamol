#include "stdafx.h"

#include "ScatterplotMatrixRenderer2D.h"
#include "FlagCall.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "vislib/math/ShallowMatrix.h"

#include <sstream>

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using vislib::sys::Log;

ScatterplotMatrixRenderer2D::ScatterplotMatrixRenderer2D() : core::view::Renderer2DModule(),
floatTableInSlot("ftIn", "Float table input"),
transferFunctionInSlot("tfIn", "Transfer function input"),
flagStorageInSlot("fsIn", "Flag storage input"),
columnsParam("columns", "Sets which columns should be displayed and in which order (empty means all)"),
colorSelectorParam("colorSelector", "Sets a color column"),
labelSelectorParam("labelSelector", "Sets a label column"),
geometryTypeParam("geometryType", "Geometry type to map data to"),
geometryWidthParam("geometryWidth", "Kernel width of the geometry, i.e., point size or line width"),
axisColorParam("axisColor", "Color of axis"),
axisWidthParam("axisWidth", "Line width for the axis"),
axisTicksXParam("axisXTicks", "Number of ticks on the X axis"),
axisTicksYParam("axisYTicks", "Number of ticks on the Y axis"),
scaleXParam("scaleX", "Aspect ratio scaling x axis length"),
scaleYParam("scaleY", "Set the scaling of y axis"),
alphaScalingParam("alphaScaling", "Scaling factor for overall alpha"),
attenuateSubpixelParam("attenuateSubpixel", "Attenuate alpha of points that should have subpixel size"),
mouseRightPressed(false), 
mouseX(0),
mouseY(0) {
    this->floatTableInSlot.SetCompatibleCall<floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->floatTableInSlot);

    this->transferFunctionInSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferFunctionInSlot);

	this->flagStorageInSlot.SetCompatibleCall<FlagCallDescription>();
	this->MakeSlotAvailable(&this->flagStorageInSlot);

	this->columnsParam << new core::param::StringParam("");
	this->MakeSlotAvailable(&this->columnsParam);
	 
    this->colorSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->colorSelectorParam);

    this->labelSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->labelSelectorParam);

    core::param::EnumParam *geometryTypes = new core::param::EnumParam(0);
	geometryTypes->SetTypePair(GEOMETRY_TYPE_POINT, "Point");
	geometryTypes->SetTypePair(GEOMETRY_TYPE_LINE, "Line");
    this->geometryTypeParam << geometryTypes;
    this->MakeSlotAvailable(&this->geometryTypeParam);

	this->geometryWidthParam << new core::param::FloatParam(1.0f, 0.0001f);
	this->MakeSlotAvailable(&this->geometryWidthParam);

	this->axisColorParam << new core::param::StringParam("white");
	this->MakeSlotAvailable(&this->axisColorParam);

	this->axisWidthParam << new core::param::FloatParam(1.0f, 0.0001f, 10000.0f);
	this->MakeSlotAvailable(&this->axisWidthParam);

    this->axisTicksXParam << new core::param::IntParam(4, 3, 100);
    this->MakeSlotAvailable(&this->axisTicksXParam);

    this->axisTicksYParam << new core::param::IntParam(4, 3, 100);
    this->MakeSlotAvailable(&this->axisTicksYParam);

    this->scaleXParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->scaleXParam);

	this->scaleYParam << new core::param::FloatParam(1.0f, 0.0001f);
	this->MakeSlotAvailable(&this->scaleYParam);

    this->alphaScalingParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->attenuateSubpixelParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->attenuateSubpixelParam);
}

ScatterplotMatrixRenderer2D::~ScatterplotMatrixRenderer2D() {
    this->Release();
}

bool ScatterplotMatrixRenderer2D::makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program) {
	vislib::graphics::gl::ShaderSource vert, frag;

	vislib::StringA vertname((prefix + "::vert").c_str());
	vislib::StringA fragname((prefix + "::frag").c_str());
	vislib::StringA pref(prefix.c_str());

	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(vertname, vert)) return false;
	if (!this->instance()->ShaderSourceFactory().MakeShaderSource(fragname, frag)) return false;

	try {
		if (!program.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile %s: Unknown error\n", pref.PeekBuffer());
			return false;
		}
	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s (@%s): %s\n", pref.PeekBuffer(),
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
				ce.FailedAction()), ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s: %s\n", pref.PeekBuffer(), e.GetMsgA());
		return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile %s: Unknown exception\n", pref.PeekBuffer());
		return false;
	}
	return true;
}

bool ScatterplotMatrixRenderer2D::create(void) {
    this->fpSeriesInsertionCB = std::bind(&ScatterplotMatrixRenderer2D::seriesInsertionCB, this, std::placeholders::_1);

    auto rw = core::utility::ResourceWrapper();
   
    // initialize OpenGL
    this->shaderInfo.ssboBindingPoint = 2;
    this->shaderInfo.numBuffers = 3;
    this->shaderInfo.bufSize = 32 * 1024 * 1024;
    this->shaderInfo.bufferCreationBits = GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT;
    this->shaderInfo.bufferMappingBits = GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT;
    this->shaderInfo.fences.resize(this->shaderInfo.numBuffers);
    this->shaderInfo.currBuf = 0;

    glGenBuffers(1, &this->shaderInfo.bufferId);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->shaderInfo.bufferId);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->shaderInfo.bufSize * this->shaderInfo.numBuffers, nullptr, this->shaderInfo.bufferCreationBits);
    this->shaderInfo.memMapPtr = glMapNamedBufferRangeEXT(this->shaderInfo.bufferId, 0, this->shaderInfo.bufSize * this->shaderInfo.numBuffers, this->shaderInfo.bufferMappingBits);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);

	if (!makeProgram("::splom", this->shaderInfo.shader)) return false;

    return true;
}

void ScatterplotMatrixRenderer2D::release(void) {
}

bool ScatterplotMatrixRenderer2D::MouseEvent(float x, float y, core::view::MouseFlags flags) {
    this->mouseX = x;
    this->mouseY = y;

    if (flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
        /*auto blubb = vislib::math::Vector<float, 3>(x, y, 1.0f);
        auto test = this->transform*blubb;

        printf("MouseCoord: %f, %f\n", test.GetX(), test.GetY());
        int i = 0;
        for (auto &r : this->bndBtns) {
            if (r.Contains(vislib::math::Point<float, 2>(test.GetX(), test.GetY()), true)) {
                this->selected[i] = !this->selected[i];
                return true;
            }
            i++;
        }*/
    } else if ((flags & core::view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)
        && this->geometryTypeParam.Param<core::param::EnumParam>()->Value() == GEOMETRY_TYPE_POINT) {
        if (flags & core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN) {
            // show tool tip for point
            this->mouseRightPressed = true;
            return true;
        } else {
            this->mouseRightPressed = false;
            return true;
        }
    }

    return false;
}

bool ScatterplotMatrixRenderer2D::Render(core::view::CallRender2D &call) {
    try {
        this->viewport = {call.GetWidth(), call.GetHeight()};

        if (!this->assertData()) return false;

        float modelViewMatrix_column[16];
        float projMatrix_column[16];

        // this is the apex of suck and must die
        glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
        glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
        // end suck

        // set NanoVG transform
        {
            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> mv(modelViewMatrix_column[0], modelViewMatrix_column[4], modelViewMatrix_column[12],
                modelViewMatrix_column[1], modelViewMatrix_column[5], modelViewMatrix_column[13],
                modelViewMatrix_column[2], modelViewMatrix_column[6], 1.0f);
            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> p(projMatrix_column[0], projMatrix_column[4], projMatrix_column[12],
                projMatrix_column[1], projMatrix_column[5], projMatrix_column[13],
                projMatrix_column[2], projMatrix_column[6], 1.0f);

            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> s1;
            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> t1;
            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> s2;

            s1.SetIdentity();
            s1.SetAt(0, 0, 1.0f);
            s1.SetAt(1, 1, -1.0f);

            t1.SetIdentity();
            t1.SetAt(0, 2, 1.f);
            t1.SetAt(1, 2, 1.f);

            s2.SetIdentity();
            s2.SetAt(0, 0, 0.5f*call.GetWidth());
            s2.SetAt(1, 1, 0.5f*call.GetHeight());

            this->nvgTrans = s2*t1*s1*p*mv;
        }

        auto diagramType = this->geometryTypeParam.Param<core::param::EnumParam>()->Value();
        switch (diagramType) {
        case GEOMETRY_TYPE_POINT:
            this->drawPoints();
            break;
		case GEOMETRY_TYPE_LINE:
			this->drawLines();
			break;
		default:
            break;
        }


    } catch (...) {
        return false;
    }

    return true;
}


bool ScatterplotMatrixRenderer2D::GetExtents(core::view::CallRender2D &call) {
    /*float w = call.GetWidth() / 2.0f;
    float h = call.GetHeight() / 2.0f;
    call.SetBoundingBox(-w, -h, w, h);*/
    call.SetBoundingBox(0.0f, 0.0f, call.GetWidth(), call.GetHeight());

    return true;
}


void ScatterplotMatrixRenderer2D::seriesInsertionCB(const DiagramSeriesCall::DiagramSeriesTuple &tuple) {
    this->columnSelectors.push_back(tuple);
}


bool ScatterplotMatrixRenderer2D::assertData(void) {
    floattable::CallFloatTableData *ft = this->floatTableInSlot.CallAs<floattable::CallFloatTableData>();
    if (ft == nullptr) return false;

    if (!(*ft)(1)) return false;
    if (!(*ft)(0)) return false;

    if (!updateColumnSelectors()) return false;

  

    if (this->dataHash == ft->DataHash() && !isAnythingDirty()) return true;

    const float *const data = ft->GetData();
    this->columnInfos = ft->GetColumnsInfos();

    const size_t rowsCount = ft->GetRowsCount();
    const size_t colCount = ft->GetColumnsCount();

    if (this->dataHash != ft->DataHash()) {
        // gather columninfo
        this->columnsParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->colorSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->labelSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        for (size_t i = 0; i < colCount; i++) {
            this->columnsParam.Param<core::param::FlexEnumParam>()->AddValue(this->columnInfos[i].Name());
            this->colorSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(this->columnInfos[i].Name());
            this->labelSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(this->columnInfos[i].Name());
        }
    }

    // set enums
    {
        auto colname = this->columnsParam.Param<core::param::FlexEnumParam>()->Value();
        this->columnIdxs.abcissaIdx = 0;
        for (size_t i = 0; i < colCount; i++) {
            if (this->columnInfos[i].Name().compare(colname) == 0)
                this->columnIdxs.abcissaIdx = i;
        }

        colname = this->colorSelectorParam.Param<core::param::FlexEnumParam>()->Value();
        this->columnIdxs.colorIdx = 0;
        for (size_t i = 0; i < colCount; i++) {
            if (this->columnInfos[i].Name().compare(colname) == 0) {
                this->columnIdxs.colorIdx = i;
                //this->minMaxColorCol.Set(ft->GetColumnsInfos()[i].MinimumValue(), ft->GetColumnsInfos()[i].MaximumValue());
            }
        }

        colname = this->labelSelectorParam.Param<core::param::FlexEnumParam>()->Value();
        this->columnIdxs.descIdx = 0;
        for (size_t i = 0; i < colCount; i++) {
            if (this->columnInfos[i].Name().compare(colname) == 0) {
                this->columnIdxs.descIdx = i;
            }
        }
    }

    // set abcissa
    {
        this->abcissa.clear();
        this->abcissa.resize(rowsCount);
        this->abcissa.shrink_to_fit();
        float abcMin = this->columnInfos[this->columnIdxs.abcissaIdx].MinimumValue();
        float abcMax = this->columnInfos[this->columnIdxs.abcissaIdx].MaximumValue();
        float abcRange = abcMax - abcMin;
        for (size_t i = 0; i < rowsCount; i++) {
            float val = (data[this->columnIdxs.abcissaIdx + i*colCount] - abcMin) / (abcRange);
            this->abcissa[i] = val; //< only valid if the selected column is sorted
        }
    }

    // set series
    {
        std::get<0>(this->yRange) = (std::numeric_limits<float>::max)();
        std::get<1>(this->yRange) = (std::numeric_limits<float>::min)();
        this->series.clear();
        this->series.resize(this->columnSelectors.size());
        this->series.shrink_to_fit();
        for (size_t s = 0; s < this->columnSelectors.size(); s++) {
            size_t seriesIdx = std::get<1>(this->columnSelectors[s]);
            this->series[s].resize(rowsCount * 4);
            this->series.shrink_to_fit();
            float minV = this->columnInfos[seriesIdx].MinimumValue();
            float maxV = this->columnInfos[seriesIdx].MaximumValue();
            float rangeV = maxV - minV;
            if (std::get<0>(this->yRange) > minV) std::get<0>(this->yRange) = minV;
            if (std::get<1>(this->yRange) < maxV) std::get<1>(this->yRange) = maxV;
            for (size_t i = 0; i < rowsCount; i++) {
                float val = (data[seriesIdx + i*colCount] - minV) / rangeV;
                this->series[s][i * 4] = this->abcissa[i];
                this->series[s][i * 4 + 1] = val;
                this->series[s][i * 4 + 2] = data[this->columnIdxs.colorIdx + i*colCount];
                this->series[s][i * 4 + 3] = data[this->columnIdxs.descIdx + i*colCount];
            }
        }
    }


    vislib::sys::Log::DefaultLog.WriteInfo("ScatterplotMatrixRenderer2D: Callstack has depth %d.\n", this->callStack.size());

    this->dataHash = ft->DataHash();
    this->resetDirtyFlag();

    return true;
}


bool ScatterplotMatrixRenderer2D::isAnythingDirty(void) const {
    return this->columnsParam.IsDirty() ||
        this->colorSelectorParam.IsDirty() ||
        this->labelSelectorParam.IsDirty();
}


void ScatterplotMatrixRenderer2D::resetDirtyFlag(void) {
    this->columnsParam.ResetDirty();
    this->colorSelectorParam.ResetDirty();
    this->labelSelectorParam.ResetDirty();
}


bool ScatterplotMatrixRenderer2D::updateColumnSelectors(void) {
//    DiagramSeriesCall *dsc = this->getColumnSelectorsSlot.CallAs<DiagramSeriesCall>();
//    if (dsc == nullptr) return false;

    std::vector<DiagramSeriesCall::DiagramSeriesTuple> oldColumnSelectors = this->columnSelectors;
    //auto oldSelected = this->selected;
    this->columnSelectors.clear();
    //this->selected.clear();

  //  dsc->SetSeriesInsertionCB(this->fpSeriesInsertionCB);
    //if (!(*dsc)(DiagramSeriesCall::CallForGetSeries)) {
    //    this->columnSelectors = oldColumnSelectors;
        //this->selected = oldSelected;
   //     return false;
   // }
    this->selectedSeries.resize(this->columnSelectors.size(), true);

    return true;
}


void ScatterplotMatrixRenderer2D::drawLines(void) {
	/*
    auto aspect = this->scaleXParam.Param<core::param::FloatParam>()->Value();
    auto yScaling = this->scaleYParam.Param<core::param::FloatParam>()->Value();

    this->drawXAxis();
    this->drawYAxis();

    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
    nvgSave(ctx);
    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));

    nvgScale(ctx, this->scaleXParam.Param<core::param::FloatParam>()->Value(), 1.0f);

    float width = std::get<0>(this->viewport)*aspect;
    float height = std::get<1>(this->viewport)*yScaling;
    float midX = width / 2.0f;
    float midY = height / 2.0f;

    for (size_t s = 0; s < this->series.size(); s++) {
        if (this->series[s].size() < 8 || !this->selectedSeries[s]) {
            continue;
        }

        auto color = std::get<4>(this->columnSelectors[s]);

        nvgStrokeWidth(ctx, this->axisWidthParam.Param<core::param::FloatParam>()->Value());
        nvgStrokeColor(ctx, nvgRGBf(color[0], color[1], color[2]));
        nvgBeginPath(ctx);

        {
            float valX = this->series[s][0] - 0.5f;
            valX *= width;
            valX += midX;
            float valY = this->series[s][1] - 0.5f;
            valY *= height;
            valY += midY;
            nvgMoveTo(ctx, valX, valY);
        }

        for (size_t i = 1; i < this->series[s].size() / 4; i++) {
            float valX = this->series[s][i * 4] - 0.5f;
            valX *= width;
            valX += midX;
            float valY = this->series[s][i * 4 + 1] - 0.5f;
            valY *= height;
            valY += midY;
            nvgLineTo(ctx, valX, valY);
        }

        nvgStroke(ctx);
    }

    nvgRestore(ctx);
	*/
}


void ScatterplotMatrixRenderer2D::drawPoints(void) {
    auto aspect = this->scaleXParam.Param<core::param::FloatParam>()->Value();
    auto yScaling = this->scaleYParam.Param<core::param::FloatParam>()->Value();

    float dw = std::get<0>(this->viewport)*aspect;
    float dh = std::get<1>(this->viewport)*yScaling;
    float mw = dw / 2.0f;
    float mh = dh / 2.0f;

    this->drawYAxis();
    this->drawXAxis();
   

    /*this->showToolTip(500, 500,
    std::string("symbol"), std::string("module"), std::string("file"), size_t(1), size_t(2), size_t(3));*/

    //TODO: Set this!!!
    float scaling = 1.0f;

    size_t idx = 0;
    if (this->mouseRightPressed) {
        idx = searchAndDispPointAttr(this->mouseX, this->mouseY);
    }

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->shaderInfo.bufferId);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->shaderInfo.ssboBindingPoint, this->shaderInfo.bufferId);

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> trans1Mat;
    trans1Mat.SetIdentity();
    trans1Mat.SetAt(0, 3, -0.5f);
    trans1Mat.SetAt(1, 3, -0.5f);
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> scaleMat;
    scaleMat.SetIdentity();
    scaleMat.SetAt(0, 0, scaling*dw);
    scaleMat.SetAt(1, 1, scaling*dh);
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> trans2Mat;
    trans2Mat.SetIdentity();
    trans2Mat.SetAt(0, 3, mw);
    trans2Mat.SetAt(1, 3, mh);
    this->oglTrans = trans2Mat*scaleMat*trans1Mat;

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    //modelViewMatrix = modelViewMatrix * scaleMat;
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelviewprojection matrix
    modelViewMatrix = modelViewMatrix*this->oglTrans;
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrixInv = modelViewMatrix;
    modelViewMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;
    // end suck

    this->shaderInfo.shader.Enable();
    //colIdxAttribLoc = glGetAttribLocation(*this->newShader, "colIdx");
    glUniform4fv(this->shaderInfo.shader.ParameterLocation("viewAttr"), 1, viewportStuff);
    //glUniform3fv(newShader->ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    //glUniform3fv(newShader->ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    //glUniform3fv(newShader->ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    //glUniform4fv(newShader->ParameterLocation("clipDat"), 1, clipDat);
    //glUniform4fv(newShader->ParameterLocation("clipCol"), 1, clipCol);
    glUniform1f(this->shaderInfo.shader.ParameterLocation("scaling"), scaling);
    glUniform1f(this->shaderInfo.shader.ParameterLocation("alphaScaling"), this->alphaScalingParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->shaderInfo.shader.ParameterLocation("attenuateSubpixel"), this->attenuateSubpixelParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);
    //glUniform1f(newShader->ParameterLocation("zNear"), cr->GetCameraParameters()->NearClip());
    glUniformMatrix4fv(this->shaderInfo.shader.ParameterLocation("modelViewProjection"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
    glUniformMatrix4fv(this->shaderInfo.shader.ParameterLocation("modelViewInverse"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
    glUniformMatrix4fv(this->shaderInfo.shader.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix.PeekComponents());
    glUniform1ui(this->shaderInfo.shader.ParameterLocation("pointIdx"), idx);
    glUniform1i(this->shaderInfo.shader.ParameterLocation("pik"), this->mouseRightPressed ? 1 : 0);

    unsigned int colTabSize = 0;
    core::view::CallGetTransferFunction *cgtf = this->transferFunctionInSlot.CallAs<core::view::CallGetTransferFunction>();
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0);
    if ((cgtf != NULL) && ((*cgtf)())) {
        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
        colTabSize = cgtf->TextureSize();
    }/* else {
     glBindTexture(GL_TEXTURE_1D, this->greyTF);
     colTabSize = 2;
     }*/
    glUniform1i(this->shaderInfo.shader.ParameterLocation("colTab"), 0);

    glUniform4f(this->shaderInfo.shader.ParameterLocation("inConsts1"), this->geometryWidthParam.Param<core::param::FloatParam>()->Value(),
        this->columnInfos[this->columnIdxs.colorIdx].MinimumValue(), this->columnInfos[this->columnIdxs.colorIdx].MaximumValue(), colTabSize);

    for (size_t s = 0; s < this->series.size(); s++) {
        // drawarrays
        size_t numElements = 4;
        size_t vertCounter = 0;
        size_t numVerts = this->shaderInfo.bufSize / (numElements * sizeof(float));
        const char *currVert = reinterpret_cast<const char *>(this->series[s].data());
        size_t numEntries = this->series[s].size() / numElements;
        //numEntries = (std::min<size_t>)(1000000, numEntries);
        while (vertCounter < numEntries) {
            void *mem = static_cast<char*>(this->shaderInfo.memMapPtr) + this->shaderInfo.bufSize * this->shaderInfo.currBuf;
            size_t vertsThisTime = vislib::math::Min(numEntries - vertCounter, numVerts);
            this->waitSingle(this->shaderInfo.fences[this->shaderInfo.currBuf]);
            memcpy(mem, currVert, vertsThisTime * numElements * sizeof(float));
            glFlushMappedNamedBufferRangeEXT(this->shaderInfo.bufferId, this->shaderInfo.bufSize * this->shaderInfo.currBuf, vertsThisTime * numElements * sizeof(float));
            //glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            //glUniform1i(this->newShader->ParameterLocation("instanceOffset"), numVerts * currBuf);
            glUniform1i(this->shaderInfo.shader.ParameterLocation("instanceOffset"), 0);
            glUniform1ui(this->shaderInfo.shader.ParameterLocation("idxOffset"), numVerts * this->shaderInfo.currBuf);

            //this->setPointers(parts, this->theSingleBuffer, reinterpret_cast<const void *>(currVert - whence), this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence));
            //glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, this->shaderInfo.ssboBindingPoint, this->shaderInfo.bufferId, this->shaderInfo.bufSize * this->shaderInfo.currBuf, this->shaderInfo.bufSize);
            glDrawArrays(GL_POINTS, 0, vertsThisTime);
            //glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
            this->lockSingle(this->shaderInfo.fences[this->shaderInfo.currBuf]);

            this->shaderInfo.currBuf = (this->shaderInfo.currBuf + 1) % this->shaderInfo.numBuffers;
            vertCounter += vertsThisTime;
            currVert += vertsThisTime * numElements * sizeof(float);
        }
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDisable(GL_TEXTURE_1D);
    this->shaderInfo.shader.Disable();
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}


void ScatterplotMatrixRenderer2D::drawXAxis(void) {
	/*
    auto numXTicks = this->axisTicksXParam.Param<core::param::IntParam>()->Value();
    auto aspect = this->scaleXParam.Param<core::param::FloatParam>()->Value();

    float minX = this->columnInfos[this->columnIdxs.abcissaIdx].MinimumValue();
    float maxX = this->columnInfos[this->columnIdxs.abcissaIdx].MaximumValue();

    std::vector<std::string> xTickText(numXTicks);
    float xTickLabel = (maxX - minX) / (numXTicks - 1);
    for (int i = 0; i < numXTicks; i++) {
        xTickText[i] = std::to_string(xTickLabel*i + minX);
    }

    float arrWidth = 0.05f;
    float arrHeight = 0.025f;
    float tickSize = this->nvgRenderInfo.fontSize;

    float dw = std::get<0>(this->viewport)*aspect;
    float dh = std::get<1>(this->viewport);
    float mw = dw / 2.0f;
    float mh = dh / 2.0f;

    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
    nvgSave(ctx);
    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));

    unsigned char col[4] = {255};
    core::utility::ColourParser::FromString(this->axisColorParam.Param<core::param::StringParam>()->Value(), 4, col);

    nvgStrokeWidth(ctx, 2.0f);
    nvgStrokeColor(ctx, nvgRGB(col[0], col[1], col[2]));

    nvgFontSize(ctx, this->nvgRenderInfo.fontSize*dh);
    nvgFontFace(ctx, "sans");
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    nvgFillColor(ctx, nvgRGB(col[0], col[1], col[2]));

    nvgBeginPath(ctx);
    nvgMoveTo(ctx, mw - dw / 2, mh - dh / 2);
    nvgLineTo(ctx, mw + (0.5f + arrWidth)*dw, mh - dh / 2);
    nvgMoveTo(ctx, mw + dw / 2, mh - (0.5f - arrHeight)*dh);
    nvgLineTo(ctx, mw + (0.5f + arrWidth)*dw, mh - dh / 2);
    nvgMoveTo(ctx, mw + dw / 2, mh - (0.5f + arrHeight)*dh);
    nvgLineTo(ctx, mw + (0.5f + arrWidth)*dw, mh - dh / 2);
    nvgStroke(ctx);

    float xTickOff = 1.0f / (numXTicks - 1);
    for (int i = 0; i < numXTicks; i++) {
        nvgBeginPath(ctx);
        nvgMoveTo(ctx, mw + (xTickOff*i - 0.5f)*dw, mh - dh / 2);
        nvgLineTo(ctx, mw + (xTickOff*i - 0.5f)*dw, mh - (0.5f + tickSize)*dh);
        nvgStroke(ctx);
    }

    nvgScale(ctx, 1.0f, -1.0f);
    nvgTranslate(ctx, 0.0f, -std::get<1>(this->viewport));

    for (int i = 0; i < numXTicks; i++) {
        nvgText(ctx, mw + (xTickOff*i - 0.5f)*dw, mh + (0.5f + tickSize)*dh, xTickText[i].c_str(), nullptr);
    }

    nvgRestore(ctx);
	*/
}


void ScatterplotMatrixRenderer2D::drawYAxis(void) {
	/*
    TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
    if (tic == nullptr) return;

    tic->SetRequest(TraceInfoCall::RequestType::GetClusterRanges, 0);
    if (!(*tic)(0)) return;

    auto clusterAddressRanges = tic->GetAddressRanges();
    auto clusterRanges = tic->GetRanges();


    int numYTicks = this->axisTicksYParam.Param<core::param::IntParam>()->Value();
    float aspect = this->scaleXParam.Param<core::param::FloatParam>()->Value();
    float yScaling = this->scaleYParam.Param<core::param::FloatParam>()->Value();

    std::vector<std::string> yTickText(numYTicks);

    float minY = std::get<0>(this->yRange);
    float maxY = std::get<1>(this->yRange);

    std::vector<float> yTicks(numYTicks);
    float yTickLabel = (maxY - minY) / (numYTicks - 1);
    float yTickOff = 1.0f / (numYTicks - 1);
    for (int i = 0; i < numYTicks; i++) {
        yTickText[i] = std::to_string(yTickLabel *i + minY);
        yTicks[i] = (1.0f / (numYTicks - 1)) * i;
    }

    float arrWidth = 0.025f;
    float arrHeight = 0.05f;
    float tickSize = this->nvgRenderInfo.fontSize*0.25f;

    float dw = std::get<0>(this->viewport); //< Should not care about aspect
    float dh = std::get<1>(this->viewport)*yScaling;
    float mw = dw / 2.0f;
    float mh = dh / 2.0f;

    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
    nvgSave(ctx);
    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));

    unsigned char col[4] = {255};
    core::utility::ColourParser::FromString(this->axisColorParam.Param<core::param::StringParam>()->Value(), 4, col);

    nvgStrokeWidth(ctx, 2.0f);
    nvgStrokeColor(ctx, nvgRGB(col[0], col[1], col[2]));

    nvgFontSize(ctx, this->nvgRenderInfo.fontSize*dh*0.5f);
    nvgFontFace(ctx, "sans");
    nvgTextAlign(ctx, NVG_ALIGN_RIGHT | NVG_ALIGN_MIDDLE);
    nvgFillColor(ctx, nvgRGB(col[0], col[1], col[2]));

    int offset = 20;

    nvgBeginPath(ctx);
    nvgMoveTo(ctx, mw - dw / 2, mh - dh / 2);
    nvgLineTo(ctx, mw - dw / 2, offset + mh + (0.5f + arrHeight)*dh);
    nvgMoveTo(ctx, mw - (0.5f + arrWidth)*dw, offset + mh + dh / 2);
    nvgLineTo(ctx, mw - dw / 2, offset + mh + (0.5f + arrHeight)*dh);
    nvgMoveTo(ctx, mw - (0.5f - arrWidth)*dw, offset + mh + dh / 2);
    nvgLineTo(ctx, mw - dw / 2, offset + mh + (0.5f + arrHeight)*dh);
    nvgStroke(ctx);

    if (clusterRanges->size() < 2) return;
	 


    // nullte
    float normY = this->columnInfos[std::get<1>(this->columnSelectors[0])].MaximumValue() - this->columnInfos[std::get<1>(this->columnSelectors[0])].MinimumValue();

    //float lastTick = mh - (0.5f - std::get<0>((*clusterRanges)[0]))*dh;
    float lastTick = (std::get<1>((*clusterRanges)[clusterRanges->size()-1]) / normY)*dh;
    nvgBeginPath(ctx);
    nvgMoveTo(ctx, mw - dw / 2, lastTick);
    nvgLineTo(ctx, mw - (0.5f + tickSize)*dw, lastTick);
    nvgStroke(ctx);

    for (size_t i = clusterRanges->size() - 2; i >= 0; i--) {
        //float currentTick = mh - (0.5f - std::get<0>((*clusterRanges)[i]))*dh;
        float currentTick = (std::get<1>((*clusterRanges)[i]) / normY)*dh;

        if (std::abs(currentTick - lastTick) > 3.0f*this->nvgRenderInfo.fontSize*dh) {
            lastTick = currentTick;
            nvgBeginPath(ctx);
            nvgMoveTo(ctx, mw - dw / 2, lastTick);
            nvgLineTo(ctx, mw - (0.5f + tickSize)*dw, lastTick);
            nvgStroke(ctx);
        }

        if (i == 0) break;
    }

    nvgScale(ctx, 1.0f, -1.0f);
    nvgTranslate(ctx, 0.0f, -std::get<1>(this->viewport));

    // nullte
    //lastTick = mh - (0.5f - std::get<0>((*clusterRanges)[0]))*dh;
    lastTick = (std::get<1>((*clusterRanges)[clusterRanges->size() - 1]) / normY)*dh;

    std::stringstream ss;
    //ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << (*clusterAddressRanges)[clusterAddressRanges->size() - 1].first;
    ss << std::hex << (*clusterAddressRanges)[clusterAddressRanges->size() - 1].first;
    nvgText(ctx, mw - (0.5f + tickSize)*dw, dh-lastTick, ss.str().c_str(), nullptr);

    for (size_t i = clusterRanges->size() - 2; i >= 0; i--) {
        //float currentTick = mh - (0.5f - std::get<0>((*clusterRanges)[i]))*dh;
        float currentTick = (std::get<1>((*clusterRanges)[i]) / normY)*dh;

        if (std::abs(currentTick - lastTick) > 3.0f*this->nvgRenderInfo.fontSize*dh) {
            lastTick = currentTick;
            std::stringstream ss;
            //ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << (*clusterAddressRanges)[i].first;
            ss << std::hex << (*clusterAddressRanges)[i].first;
            nvgText(ctx, mw - (0.5f + tickSize)*dw, dh - lastTick, ss.str().c_str(), nullptr);
        }

        if (i == 0) break;
    }

    nvgRestore(ctx);
	*/
}

  


void ScatterplotMatrixRenderer2D::drawToolTip(const float x, const float y, const std::string &text) const {
    /*auto ctx = static_cast<NVGcontext *>(this->nvgCtx);
    float ttOH = 10;
    float ttOW = 10;
    float ttW = 200;
    float ttH = ttOH + 6 * (ttOH + BND_WIDGET_HEIGHT);

    float heightOffset = ttOH + BND_WIDGET_HEIGHT;


    nvgBeginPath(ctx);
    nvgFontSize(ctx, 10.0f);
    nvgFillColor(ctx, nvgRGB(128, 128, 128));

    nvgRect(ctx, x, y, ttW, ttH);

    //bndTooltipBackground(ctx, x, y, ttW, ttH);
    bndTextField(ctx, x + ttOW, y + ttOH, ttW - 2 * ttOW, ttH, BND_CORNER_ALL, BND_DEFAULT, -1, text.c_str(), 0, text.size() - 1);

    nvgFill(ctx);
	*/
}


size_t ScatterplotMatrixRenderer2D::searchAndDispPointAttr(const float x, const float y) {
	/*
    if (y >= 0.0f) { //< within scatterplot
        auto trans = this->oglTrans;
        trans.Invert();
        auto query_p = trans*vislib::math::Vector<float, 4>(x, y, 0.0f, 1.0f);
        float qp[2] = {query_p.X(), query_p.Y()};
        // search with nanoflann tree
        size_t idx[1] = {0};
        float dis[1] = {0.0f};
        this->tree->index->knnSearch(qp, 1, idx, dis);

        idx[0] = *reinterpret_cast<unsigned int *>(&this->series[0][idx[0] * 4 + 3]); //< toxic, which is the correct series?

        auto ssp = this->nvgTrans*vislib::math::Vector<float, 3>(x, y, 1.0f);
        TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
        if (tic == nullptr) {
            // show tool tip
            this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Call"));
        } else {
            tic->SetRequest(TraceInfoCall::RequestType::GetSymbolString, idx[0]);
            if (!(*tic)(0)) {
                // show tool tip
                this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Found"));
            } else {
                auto st = tic->GetInfo();
                this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, st);
            }
        }

        return idx[0];
    } else { //< within callstack
        // calculate depth
        // search for fitting range in chosen depth
        float boxHeight = std::get<1>(this->viewport) / 40.0f;
        float yCoord = std::fabsf(y);
        unsigned int depth = std::floorf(yCoord / boxHeight);
        auto ssp = this->nvgTrans*vislib::math::Vector<float, 3>(x, y, 1.0f);
        float aspect = this->scaleXParam.Param<core::param::FloatParam>()->Value();
        for (auto &r : this->callStack[depth]) {
            // rb / norm*dw
            float rb = std::get<0>(r);
            float re = std::get<1>(r);
            if ((rb / this->abcissa.size()*std::get<0>(this->viewport)*aspect) <= x && x <= (re / this->abcissa.size()*std::get<0>(this->viewport)*aspect)) { //< abcissa missing
                size_t symbolIdx = std::get<2>(r);
                TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
                if (tic == nullptr) {
                    // show tool tip
                    this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Call"));
                } else {
                    tic->SetRequest(TraceInfoCall::RequestType::GetSymbolString, symbolIdx);
                    if (!(*tic)(0)) {
                        // show tool tip
                        this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Found"));
                    } else {
                        auto st = tic->GetInfo();
                        st += std::string(" ") + std::to_string((unsigned int)rb) + std::string(" ") + std::to_string((unsigned int)re);
                        this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, st);
                    }
                }
                return symbolIdx;
            }
        }
    }
	*/
    return 0;
}


void ScatterplotMatrixRenderer2D::lockSingle(GLsync &syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


void ScatterplotMatrixRenderer2D::waitSingle(GLsync &syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}
