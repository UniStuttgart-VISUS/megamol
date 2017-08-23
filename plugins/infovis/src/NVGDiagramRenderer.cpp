#include "stdafx.h"
#include "NVGDiagramRenderer.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "vislib/math/ShallowMatrix.h"

#include "nanovg.h"
#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg_gl.h"

#define BLENDISH_IMPLEMENTATION
#include "blendish.h"

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using vislib::sys::Log;


NVGDiagramRenderer::NVGDiagramRenderer() : core::view::Renderer2DModule(),
floatTableInSlot("ftIn", "FloatTable input"),
getColumnSelectorsSlot("columnSelIn", "Slot asking for selected columns"),
getTransFuncSlot("transFuncIn", "Slot for transfer function"),
//getCallTraceSlot("callTraceIn", "Slot for call trace"),
//getPointInfoSlot("pointInfoIn", "Slot for point info requests"),
abcissaSelectParam("abcissaSelector", "Slot to select column as abcissa"),
colorSelectParam("colorSelector", "Param to select color column"),
descSelectParam("descSelector", "Param to select desc column"),
diagramTypeParam("Type", "The diagram type"),
lineWidthParam("lineWidth", "Sets the line width"),
numXTicksParam("X Ticks", "The number of X ticks"),
numYTicksParam("Y Ticks", "The number of Y ticks"),
aspectParam("aspect", "Aspect ratio scaling x axis length"),
alphaScalingParam("alphaScaling", "scaling factor for particle alpha"),
attenuateSubpixelParam("attenuateSubpixel", "attenuate alpha of points that should have subpixel size"),
pointSizeParam("pointSize", "Set the size of points in scatterplot"),
yScalingParam("yScaling", "Set the scaling of y axis"),
textThresholdParam("textThreshold", "Set the threshold for text rendering"),
pyjamaModeParam("toggle pyjama", "Toggles pyjama mode of scatterplot"),
mouseRightPressed(false), mouseX(0), mouseY(0) {
    this->floatTableInSlot.SetCompatibleCall<floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->floatTableInSlot);

    this->getColumnSelectorsSlot.SetCompatibleCall<DiagramSeriesCallDescription>();
    this->MakeSlotAvailable(&this->getColumnSelectorsSlot);

    this->getTransFuncSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTransFuncSlot);

    /*this->getCallTraceSlot.SetCompatibleCall<CallTraceCallDescription>();
    this->MakeSlotAvailable(&this->getCallTraceSlot);

    this->getPointInfoSlot.SetCompatibleCall<TraceInfoCallDescription>();
    this->MakeSlotAvailable(&this->getPointInfoSlot);*/

    core::param::FlexEnumParam *abcissaSelectorEP = new core::param::FlexEnumParam("undef");
    this->abcissaSelectParam << abcissaSelectorEP;
    this->MakeSlotAvailable(&this->abcissaSelectParam);

    core::param::FlexEnumParam *colorSelectorEP = new core::param::FlexEnumParam("undef");
    this->colorSelectParam << colorSelectorEP;
    this->MakeSlotAvailable(&this->colorSelectParam);

    core::param::FlexEnumParam *descSelectorEP = new core::param::FlexEnumParam("undef");
    this->descSelectParam << descSelectorEP;
    this->MakeSlotAvailable(&this->descSelectParam);

    core::param::EnumParam *dt = new core::param::EnumParam(0);
    dt->SetTypePair(DIAGRAM_TYPE_LINE_PLOT, "Line");
    dt->SetTypePair(DIAGRAM_TYPE_SCATTER_PLOT, "Scatterplot");
    this->diagramTypeParam << dt;
    this->MakeSlotAvailable(&this->diagramTypeParam);

    this->numXTicksParam << new core::param::IntParam(4, 3, 100);
    this->MakeSlotAvailable(&this->numXTicksParam);

    this->numYTicksParam << new core::param::IntParam(4, 3, 100);
    this->MakeSlotAvailable(&this->numYTicksParam);

    this->lineWidthParam << new core::param::FloatParam(1.0f, 0.0001f, 10000.0f);
    this->MakeSlotAvailable(&this->lineWidthParam);

    this->aspectParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->aspectParam);

    this->alphaScalingParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->attenuateSubpixelParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->attenuateSubpixelParam);

    this->pointSizeParam << new core::param::FloatParam(1.0f, 0.0001f);
    this->MakeSlotAvailable(&this->pointSizeParam);

    this->yScalingParam << new core::param::FloatParam(1.0f, 0.0001f);
    this->MakeSlotAvailable(&this->yScalingParam);

    this->textThresholdParam << new core::param::IntParam(200, 0);
    this->MakeSlotAvailable(&this->textThresholdParam);

    this->pyjamaModeParam << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->pyjamaModeParam);
}


NVGDiagramRenderer::~NVGDiagramRenderer() {
    this->Release();
}


bool NVGDiagramRenderer::create(void) {
    this->fpSeriesInsertionCB = std::bind(&NVGDiagramRenderer::seriesInsertionCB, this, std::placeholders::_1);

    this->nvgCtx = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
    if (this->nvgCtx == nullptr) {
        Log::DefaultLog.WriteError("NVGDiagramRenderer: Could not init NanoVG\n");
        return false;
    }

    auto rw = core::utility::ResourceWrapper();
    /*char *filepaths = NULL;
    size_t outSize = rw.LoadTextResource(this->GetCoreInstance()->Configuration(), vislib::StringA("infovis_filepaths.txt"), &filepaths);*/
    auto font_path = rw.getFileName(this->GetCoreInstance()->Configuration(), vislib::StringA("DejaVuSans.ttf"));
    auto icon_path = rw.getFileName(this->GetCoreInstance()->Configuration(), vislib::StringA("blender_icons.svg"));

    /*if (filepaths == NULL) {
        return NULL;
    }

    vislib::StringA font_path;
    vislib::StringA icon_path;

    auto tokenizer = vislib::StringTokeniserA(filepaths, "\n");
    if (tokenizer.HasNext()) {
        font_path = tokenizer.Next();
        font_path.Trim(" \r");
    } else {
        return false;
    }

    if (tokenizer.HasNext()) {
        icon_path = tokenizer.Next();
        icon_path.Trim(" \r");
    } else {
        return false;
    }*/

    auto nvgFontSans = nvgCreateFont((NVGcontext*)this->nvgCtx, "sans", std::string(W2A(font_path)).c_str());
    bndSetFont(nvgFontSans);
    bndSetIconImage(nvgCreateImage((NVGcontext*)this->nvgCtx, std::string(W2A(icon_path)).c_str(), 0));

    this->nvgRenderInfo.fontSize = 1.0f / 20.0f;

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

    char *vShader = NULL;
    size_t vShaderS = 0;
    char *fShader = NULL;
    size_t fShaderS = 0;

    vShaderS = rw.LoadTextResource(this->GetCoreInstance()->Configuration(), "nvgdr2_splat_v.glsl", &vShader);
    fShaderS = rw.LoadTextResource(this->GetCoreInstance()->Configuration(), "nvgdr2_splat_f.glsl", &fShader);

    try {
        if (!this->shaderInfo.shader.Create(vShader, fShader)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return nullptr;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return nullptr;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return nullptr;
    }

    delete[] vShader, fShader;

    return true;
}


void NVGDiagramRenderer::release(void) {
    //SAFE_DELETE(this->tree);
}


bool NVGDiagramRenderer::MouseEvent(float x, float y, core::view::MouseFlags flags) {
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
        && this->diagramTypeParam.Param<core::param::EnumParam>()->Value() == DIAGRAM_TYPE_SCATTER_PLOT) {
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


bool NVGDiagramRenderer::Render(core::view::CallRender2D &call) {
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

        nvgBeginFrame(static_cast<NVGcontext *>(this->nvgCtx), std::get<0>(this->viewport), std::get<1>(this->viewport), 1.0f);

        auto diagramType = this->diagramTypeParam.Param<core::param::EnumParam>()->Value();
        switch (diagramType) {
        case DIAGRAM_TYPE_LINE_PLOT:
            this->drawLinePlot();
            break;
        case DIAGRAM_TYPE_SCATTER_PLOT:
            this->drawScatterPlot();
            break;
        default:
            break;
        }

        nvgEndFrame(static_cast<NVGcontext *>(this->nvgCtx));
    } catch (...) {
        return false;
    }

    return true;
}


bool NVGDiagramRenderer::GetExtents(core::view::CallRender2D &call) {
    /*float w = call.GetWidth() / 2.0f;
    float h = call.GetHeight() / 2.0f;
    call.SetBoundingBox(-w, -h, w, h);*/
    call.SetBoundingBox(0.0f, 0.0f, call.GetWidth(), call.GetHeight());

    return true;
}


void NVGDiagramRenderer::seriesInsertionCB(const DiagramSeriesCall::DiagramSeriesTuple &tuple) {
    this->columnSelectors.push_back(tuple);
}


bool NVGDiagramRenderer::assertData(void) {
    floattable::CallFloatTableData *ft = this->floatTableInSlot.CallAs<floattable::CallFloatTableData>();
    if (ft == nullptr) return false;

    if (!(*ft)(1)) return false;
    if (!(*ft)(0)) return false;

    if (!updateColumnSelectors()) return false;

    /*CallTraceCall *ctc = this->getCallTraceSlot.CallAs<CallTraceCall>();
    if (ctc == nullptr) return false;

    if (!(*ctc)(0)) return false;*/

    if (this->dataHash == ft->DataHash() && !isAnythingDirty()) return true;

    const float *const data = ft->GetData();
    this->columnInfos = ft->GetColumnsInfos();

    const size_t rowsCount = ft->GetRowsCount();
    const size_t colCount = ft->GetColumnsCount();

    if (this->dataHash != ft->DataHash()) {
        // gather columninfo
        this->abcissaSelectParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->colorSelectParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->descSelectParam.Param<core::param::FlexEnumParam>()->ClearValues();
        for (size_t i = 0; i < colCount; i++) {
            this->abcissaSelectParam.Param<core::param::FlexEnumParam>()->AddValue(this->columnInfos[i].Name());
            this->colorSelectParam.Param<core::param::FlexEnumParam>()->AddValue(this->columnInfos[i].Name());
            this->descSelectParam.Param<core::param::FlexEnumParam>()->AddValue(this->columnInfos[i].Name());
        }
    }

    // set enums
    {
        auto colname = this->abcissaSelectParam.Param<core::param::FlexEnumParam>()->Value();
        this->columnIdxs.abcissaIdx = 0;
        for (size_t i = 0; i < colCount; i++) {
            if (this->columnInfos[i].Name().compare(colname) == 0)
                this->columnIdxs.abcissaIdx = i;
        }

        colname = this->colorSelectParam.Param<core::param::FlexEnumParam>()->Value();
        this->columnIdxs.colorIdx = 0;
        for (size_t i = 0; i < colCount; i++) {
            if (this->columnInfos[i].Name().compare(colname) == 0) {
                this->columnIdxs.colorIdx = i;
                //this->minMaxColorCol.Set(ft->GetColumnsInfos()[i].MinimumValue(), ft->GetColumnsInfos()[i].MaximumValue());
            }
        }

        colname = this->descSelectParam.Param<core::param::FlexEnumParam>()->Value();
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

    // create call stack
    //{
    //    auto callstacklist = ctc->GetCallTrace();
    //    this->callStack.clear();
    //    size_t maxCallStackDepth = 0;
    //    for (size_t instr = 0; instr < callstacklist->size(); instr++) {
    //        if (maxCallStackDepth < (*callstacklist)[instr].size()) maxCallStackDepth = (*callstacklist)[instr].size();
    //    }
    //    if (maxCallStackDepth > 0) {
    //        this->callStack.resize(maxCallStackDepth);
    //        for (size_t depth = 0; depth < maxCallStackDepth; depth++) {
    //            size_t currSymbol = 0;
    //            call_t currRange = {0.0f, 0.0f, currSymbol};
    //            for (size_t instr = 0; instr < callstacklist->size(); instr++) {
    //                if (depth >= (*callstacklist)[instr].size()) {
    //                    // complete current call
    //                    std::get<1>(currRange) = instr;
    //                    std::get<2>(currRange) = currSymbol;
    //                    if (currSymbol != 0) {
    //                        // push call
    //                        this->callStack[depth].push_back(currRange);
    //                    }
    //                    currRange = {std::get<1>(currRange), std::get<1>(currRange), 0};
    //                    currSymbol = 0;
    //                } else {
    //                    if (currSymbol != (*callstacklist)[instr][depth]) {
    //                        // complete current call
    //                        std::get<1>(currRange) = instr;
    //                        std::get<2>(currRange) = currSymbol;
    //                        if (currSymbol != 0) {
    //                            // push call
    //                            this->callStack[depth].push_back(currRange);
    //                        }
    //                        currSymbol = (*callstacklist)[instr][depth];
    //                        currRange = {std::get<1>(currRange), std::get<1>(currRange), currSymbol};
    //                    }
    //                }
    //            }
    //            if (currSymbol != 0) {
    //                std::get<1>(currRange) = callstacklist->size();
    //                this->callStack[depth].push_back(currRange);
    //            }
    //        }
    //    }
    //}

    // create kdtree
    //this->tree = new kdTree_t(this->series, 0); //< toxic, set index properly

    vislib::sys::Log::DefaultLog.WriteInfo("NVGDiagramRenderer: Callstack has depth %d.\n", this->callStack.size());

    this->dataHash = ft->DataHash();
    this->resetDirtyFlag();

    return true;
}


bool NVGDiagramRenderer::isAnythingDirty(void) const {
    return this->abcissaSelectParam.IsDirty() ||
        this->colorSelectParam.IsDirty() ||
        this->descSelectParam.IsDirty();
}


void NVGDiagramRenderer::resetDirtyFlag(void) {
    this->abcissaSelectParam.ResetDirty();
    this->colorSelectParam.ResetDirty();
    this->descSelectParam.ResetDirty();
}


bool NVGDiagramRenderer::updateColumnSelectors(void) {
    DiagramSeriesCall *dsc = this->getColumnSelectorsSlot.CallAs<DiagramSeriesCall>();
    if (dsc == nullptr) return false;

    std::vector<DiagramSeriesCall::DiagramSeriesTuple> oldColumnSelectors = this->columnSelectors;
    //auto oldSelected = this->selected;
    this->columnSelectors.clear();
    //this->selected.clear();

    dsc->SetSeriesInsertionCB(this->fpSeriesInsertionCB);
    if (!(*dsc)(DiagramSeriesCall::CallForGetSeries)) {
        this->columnSelectors = oldColumnSelectors;
        //this->selected = oldSelected;
        return false;
    }
    this->selectedSeries.resize(this->columnSelectors.size(), true);

    return true;
}


void NVGDiagramRenderer::drawLinePlot(void) {
    auto aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();
    auto yScaling = this->yScalingParam.Param<core::param::FloatParam>()->Value();

    this->drawXAxis();
    this->drawYAxis();

    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
    nvgSave(ctx);
    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));

    nvgScale(ctx, this->aspectParam.Param<core::param::FloatParam>()->Value(), 1.0f);

    float width = std::get<0>(this->viewport)*aspect;
    float height = std::get<1>(this->viewport)*yScaling;
    float midX = width / 2.0f;
    float midY = height / 2.0f;

    for (size_t s = 0; s < this->series.size(); s++) {
        if (this->series[s].size() < 8 || !this->selectedSeries[s]) {
            continue;
        }

        auto color = std::get<4>(this->columnSelectors[s]);

        nvgStrokeWidth(ctx, this->lineWidthParam.Param<core::param::FloatParam>()->Value());
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
}


void NVGDiagramRenderer::drawScatterPlot(void) {
    auto aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();
    auto yScaling = this->yScalingParam.Param<core::param::FloatParam>()->Value();

    float dw = std::get<0>(this->viewport)*aspect;
    float dh = std::get<1>(this->viewport)*yScaling;
    float mw = dw / 2.0f;
    float mh = dh / 2.0f;

    //if (this->pyjamaModeParam.Param<core::param::BoolParam>()->Value()) {
    //    //this->drawXAxis();
    //    this->drawPyjama();
    //} /*else {
    //  this->drawXAxis();
    //  }*/
    this->drawYAxis();
    //this->drawCallStack();

    /*this->showToolTip(500, 500,
    std::string("symbol"), std::string("module"), std::string("file"), size_t(1), size_t(2), size_t(3));*/

    //TODO: Set this!!!
    float scaling = 1.0f;

    /*size_t idx = 0;
    if (this->mouseRightPressed) {
        idx = searchAndDispPointAttr(this->mouseX, this->mouseY);
    }*/

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
    /*glUniform1ui(this->shaderInfo.shader.ParameterLocation("pointIdx"), idx);
    glUniform1i(this->shaderInfo.shader.ParameterLocation("pik"), this->mouseRightPressed ? 1 : 0);*/

    unsigned int colTabSize = 0;
    core::view::CallGetTransferFunction *cgtf = this->getTransFuncSlot.CallAs<core::view::CallGetTransferFunction>();
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

    glUniform4f(this->shaderInfo.shader.ParameterLocation("inConsts1"), this->pointSizeParam.Param<core::param::FloatParam>()->Value(),
        this->columnInfos[this->columnIdxs.colorIdx].MinimumValue(), this->columnInfos[this->columnIdxs.colorIdx].MaximumValue(), colTabSize);

    for (size_t s = 0; s < this->series.size(); s++) {
        // drawarrays
        size_t numElements = 4;
        size_t vertCounter = 0;
        size_t numVerts = this->shaderInfo.bufSize / (numElements * sizeof(float));
        const char *currVert = reinterpret_cast<const char *>(this->series[s].data());
        size_t numEntries = this->series[s].size() / numElements;
        numEntries = (std::min<size_t>)(1000000, numEntries);
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


void NVGDiagramRenderer::drawXAxis(void) {
    auto numXTicks = this->numXTicksParam.Param<core::param::IntParam>()->Value();
    auto aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();

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

    nvgStrokeWidth(ctx, 2.0f);
    nvgStrokeColor(ctx, nvgRGB(255, 255, 255));

    nvgFontSize(ctx, this->nvgRenderInfo.fontSize*dh);
    nvgFontFace(ctx, "sans");
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    nvgFillColor(ctx, nvgRGB(255, 255, 255));

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
}


void NVGDiagramRenderer::drawYAxis(void) {
    int numYTicks = this->numYTicksParam.Param<core::param::IntParam>()->Value();
    float aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();
    float yScaling = this->yScalingParam.Param<core::param::FloatParam>()->Value();

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
    float tickSize = this->nvgRenderInfo.fontSize;

    float dw = std::get<0>(this->viewport); //< Should not care about aspect
    float dh = std::get<1>(this->viewport)*yScaling;
    float mw = dw / 2.0f;
    float mh = dh / 2.0f;

    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
    nvgSave(ctx);
    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));

    nvgStrokeWidth(ctx, 2.0f);
    nvgStrokeColor(ctx, nvgRGB(255, 255, 255));

    nvgFontSize(ctx, this->nvgRenderInfo.fontSize*dh);
    nvgFontFace(ctx, "sans");
    nvgTextAlign(ctx, NVG_ALIGN_RIGHT | NVG_ALIGN_MIDDLE);
    nvgFillColor(ctx, nvgRGB(255, 255, 255));

    nvgBeginPath(ctx);
    nvgMoveTo(ctx, mw - dw / 2, mh - dh / 2);
    nvgLineTo(ctx, mw - dw / 2, mh + (0.5f + arrHeight)*dh);
    nvgMoveTo(ctx, mw - (0.5f + arrWidth)*dw, mh + dh / 2);
    nvgLineTo(ctx, mw - dw / 2, mh + (0.5f + arrHeight)*dh);
    nvgMoveTo(ctx, mw - (0.5f - arrWidth)*dw, mh + dh / 2);
    nvgLineTo(ctx, mw - dw / 2, mh + (0.5f + arrHeight)*dh);
    nvgStroke(ctx);

    for (int i = 0; i < numYTicks; i++) {
        nvgBeginPath(ctx);
        nvgMoveTo(ctx, mw - dw / 2, mh - (0.5f - yTicks[i])*dh);
        nvgLineTo(ctx, mw - (0.5f + tickSize)*dw, mh - (0.5f - yTicks[i])*dh);
        nvgStroke(ctx);
    }

    nvgScale(ctx, 1.0f, -1.0f);
    nvgTranslate(ctx, 0.0f, -std::get<1>(this->viewport));
    for (int i = 0; i < numYTicks; i++) {
        nvgText(ctx, mw - (0.5f + tickSize)*dw, mh - (0.5f - yTicks[i])*dh, yTickText[numYTicks - 1 - i].c_str(), nullptr);
    }

    nvgRestore(ctx);
}


//void NVGDiagramRenderer::drawCallStack(void) {
//    TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
//    if (tic == nullptr) return;
//
//    unsigned int colTabSize = 0;
//    const float *texture = NULL;
//    unsigned int numElements = 0;
//    core::view::CallGetTransferFunction *cgtf = this->getTransFuncSlot.CallAs<core::view::CallGetTransferFunction>();
//    if ((cgtf != NULL) && ((*cgtf)())) {
//        texture = cgtf->GetTextureData();
//        colTabSize = cgtf->TextureSize();
//        switch (cgtf->OpenGLTextureFormat())
//        {
//        case core::view::CallGetTransferFunction::TEXTURE_FORMAT_RGB:
//            numElements = 3;
//            break;
//        case core::view::CallGetTransferFunction::TEXTURE_FORMAT_RGBA:
//            numElements = 4;
//            break;
//        default:
//            break;
//        }
//    } else return;
//    auto minColorIdx = this->columnInfos[this->columnIdxs.colorIdx].MinimumValue();
//    auto maxColorIdx = this->columnInfos[this->columnIdxs.colorIdx].MaximumValue();
//
//    float aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();
//
//    float dw = std::get<0>(this->viewport)*aspect;
//    float dh = std::get<1>(this->viewport);
//    float mw = dw / 2.0f;
//    float mh = dh / 2.0f;
//
//    float boxHeight = dh / 40.0f;
//
//    float norm = this->abcissa.size();
//
//    int textThreshold = this->textThresholdParam.Param<core::param::IntParam>()->Value();
//
//    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
//    nvgSave(ctx);
//    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
//        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
//        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));
//    nvgScale(ctx, 1.0f, -1.0f);
//
//    nvgFontSize(ctx, this->nvgRenderInfo.fontSize*dh*0.5f);
//    nvgFontFace(ctx, "sans");
//    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
//
//    nvgFillColor(ctx, nvgRGB(255, 255, 255));
//
//    nvgBeginPath(ctx);
//
//    for (size_t depth = 0; depth < this->callStack.size(); depth++) {
//        for (auto &r : this->callStack[depth]) {
//            float rb = std::get<0>(r);
//            float re = std::get<1>(r);
//            float rwidth = re - rb;
//
//            size_t symbolIdx = std::get<2>(r);
//            tic->SetRequest(TraceInfoCall::RequestType::GetModuleColor, symbolIdx);
//            if ((*tic)(0)) {
//                auto colorIdx = tic->GetColorIdx();
//                // query transferfunction
//                float cid = ((float)colorIdx - minColorIdx) / (maxColorIdx - minColorIdx);
//                cid *= (1.0f - 1.0f / colTabSize);
//                cid += 0.5f / colTabSize;
//
//                float midIdx = colTabSize*cid;
//                unsigned int minIdx = std::floorf(midIdx);
//                unsigned int maxIdx = std::ceilf(midIdx);
//
//                float inter = midIdx - minIdx;
//
//                nvgFillColor(ctx, nvgRGBf(texture[minIdx*numElements] * (1.0f - inter) + texture[maxIdx*numElements] * inter,
//                    texture[minIdx*numElements + 1] * (1.0f - inter) + texture[maxIdx*numElements + 1] * inter,
//                    texture[minIdx*numElements + 2] * (1.0f - inter) + texture[maxIdx*numElements + 2] * inter));
//            }
//
//            nvgRect(ctx, rb / norm*dw, boxHeight*depth, rwidth / norm*dw, boxHeight);
//            nvgFill(ctx);
//            if (rwidth / norm*dw > textThreshold) { //< Are there only global color settings in NanoVG? That would be pretty bad.
//                auto text = tic->GetInfo();
//                nvgFillColor(ctx, nvgRGB(255, 255, 255));
//                nvgText(ctx, (rb + rwidth / 2) / norm*dw, boxHeight*depth, text.c_str(), nullptr);
//            }
//            nvgBeginPath(ctx);
//        }
//    }
//
//    nvgRestore(ctx);
//}


//void NVGDiagramRenderer::drawPyjama(void) {
//    TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
//    if (tic == nullptr) return;
//
//    tic->SetRequest(TraceInfoCall::RequestType::GetClusterRanges, 0);
//    if (!(*tic)(0)) return;
//
//    auto clusterRanges = tic->GetRanges();
//
//    float aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();
//    float yScaling = this->yScalingParam.Param<core::param::FloatParam>()->Value();
//    float dw = std::get<0>(this->viewport)*aspect;
//    float dh = std::get<1>(this->viewport)*yScaling;
//    float mw = dw / 2.0f;
//    float mh = dh / 2.0f;
//
//    float normY = this->columnInfos[std::get<1>(this->columnSelectors[0])].MaximumValue() - this->columnInfos[std::get<1>(this->columnSelectors[0])].MinimumValue();
//
//    NVGcontext *ctx = static_cast<NVGcontext *>(this->nvgCtx);
//    nvgSave(ctx);
//    nvgTransform(ctx, this->nvgTrans.GetAt(0, 0), this->nvgTrans.GetAt(1, 0),
//        this->nvgTrans.GetAt(0, 1), this->nvgTrans.GetAt(1, 1),
//        this->nvgTrans.GetAt(0, 2), this->nvgTrans.GetAt(1, 2));
//
//    nvgFillColor(ctx, nvgRGBA(255, 255, 255, 128));
//
//    nvgBeginPath(ctx);
//
//    for (size_t i = 0; i < clusterRanges->size(); i++) {
//        auto range = (*clusterRanges)[i];
//        nvgRect(ctx, 0, (std::get<0>(range) / normY)*dh, dw, ((std::get<1>(range) - std::get<0>(range)) / normY)*dh);
//    }
//
//    nvgFill(ctx);
//
//    nvgRestore(ctx);
//}


//void NVGDiagramRenderer::drawToolTip(const float x, const float y, const std::string &text) const {
//    auto ctx = static_cast<NVGcontext *>(this->nvgCtx);
//    float ttOH = 10;
//    float ttOW = 10;
//    float ttW = 200;
//    float ttH = ttOH + 6 * (ttOH + BND_WIDGET_HEIGHT);
//
//    float heightOffset = ttOH + BND_WIDGET_HEIGHT;
//
//
//    nvgBeginPath(ctx);
//    nvgFontSize(ctx, 10.0f);
//    nvgFillColor(ctx, nvgRGB(128, 128, 128));
//
//    nvgRect(ctx, x, y, ttW, ttH);
//
//    //bndTooltipBackground(ctx, x, y, ttW, ttH);
//    bndTextField(ctx, x + ttOW, y + ttOH, ttW - 2 * ttOW, ttH, BND_CORNER_ALL, BND_DEFAULT, -1, text.c_str(), 0, text.size() - 1);
//
//    nvgFill(ctx);
//}


//size_t NVGDiagramRenderer::searchAndDispPointAttr(const float x, const float y) {
//    if (y >= 0.0f) { //< within scatterplot
//        auto trans = this->oglTrans;
//        trans.Invert();
//        auto query_p = trans*vislib::math::Vector<float, 4>(x, y, 0.0f, 1.0f);
//        float qp[2] = {query_p.X(), query_p.Y()};
//        // search with nanoflann tree
//        size_t idx[1] = {0};
//        float dis[1] = {0.0f};
//        this->tree->index->knnSearch(qp, 1, idx, dis);
//
//        idx[0] = *reinterpret_cast<unsigned int *>(&this->series[0][idx[0] * 4 + 3]); //< toxic, which is the correct series?
//
//        auto ssp = this->nvgTrans*vislib::math::Vector<float, 3>(x, y, 1.0f);
//        TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
//        if (tic == nullptr) {
//            // show tool tip
//            this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Call"));
//        } else {
//            tic->SetRequest(TraceInfoCall::RequestType::GetSymbolString, idx[0]);
//            if (!(*tic)(0)) {
//                // show tool tip
//                this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Found"));
//            } else {
//                auto st = tic->GetInfo();
//                this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, st);
//            }
//        }
//
//        return idx[0];
//    } else { //< within callstack
//             // calculate depth
//             // search for fitting range in chosen depth
//        float boxHeight = std::get<1>(this->viewport) / 40.0f;
//        float yCoord = std::fabsf(y);
//        unsigned int depth = std::floorf(yCoord / boxHeight);
//        auto ssp = this->nvgTrans*vislib::math::Vector<float, 3>(x, y, 1.0f);
//        float aspect = this->aspectParam.Param<core::param::FloatParam>()->Value();
//        for (auto &r : this->callStack[depth]) {
//            // rb / norm*dw
//            float rb = std::get<0>(r);
//            float re = std::get<1>(r);
//            if ((rb / this->abcissa.size()*std::get<0>(this->viewport)*aspect) <= x && x <= (re / this->abcissa.size()*std::get<0>(this->viewport)*aspect)) { //< abcissa missing
//                size_t symbolIdx = std::get<2>(r);
//                TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
//                if (tic == nullptr) {
//                    // show tool tip
//                    this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Call"));
//                } else {
//                    tic->SetRequest(TraceInfoCall::RequestType::GetSymbolString, symbolIdx);
//                    if (!(*tic)(0)) {
//                        // show tool tip
//                        this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Found"));
//                    } else {
//                        auto st = tic->GetInfo();
//                        this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, st);
//                    }
//                }
//                return symbolIdx;
//            }
//        }
//    }
//
//    return 0;
//}


void NVGDiagramRenderer::lockSingle(GLsync &syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


void NVGDiagramRenderer::waitSingle(GLsync &syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}
