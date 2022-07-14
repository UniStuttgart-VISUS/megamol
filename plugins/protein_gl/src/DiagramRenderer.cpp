#include "DiagramRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "vislib/graphics/PngBitmapCodec.h"
#include "vislib/math/Rectangle.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"
#include "vislib_gl/graphics/gl/Verdana.inc"
#include <float.h>
#include <math.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_gl;
using namespace vislib_gl::graphics::gl;
using megamol::core::utility::log::Log;

/*
 * DiagramRenderer::DiagramRenderer (CTOR)
 */
DiagramRenderer::DiagramRenderer(void)
        : mmstd_gl::Renderer2DModuleGL()
        , dataCallerSlot("getData", "Connects the diagram rendering with data storage.")
        , selectionCallerSlot("getSelection", "Connects the diagram rendering with selection storage.")
        , hiddenCallerSlot("getHidden", "Connects the diagram rendering with visibility storage.")
        , theFont(FontInfo_Verdana)
        , decorationDepth(0.0f)
        , diagramTypeParam("Type", "The diagram type.")
        , diagramStyleParam("Style", "The diagram style.")
        , numXTicksParam("X Ticks", "The number of X ticks.")
        , numYTicksParam("Y Ticks", "The number of Y ticks.")
        , lineWidthParam("linewidth", "width of the drawn lines.")
        , drawYLogParam("logarithmic", "scale the Y axis logarithmically.")
        , foregroundColorParam("foregroundCol", "The color of the diagram decorations.")
        , drawCategoricalParam("categorical", "Draw column charts as categorical.")
        , showCrosshairParam("show XHair", "bool param for the Crosshair toggle")
        , showCrosshairToggleParam("toggle XHair", "Show a crosshair to inform the user of the current mouse position.")
        , showAllParam("show all", "Make all series visible.")
        , hideAllParam("hide all", "Make all series invisible.")
        , showGuidesParam("show guides", "Show defined guides in the diagram.")
        , autoAspectParam("auto aspect", "Automatically adjust aspect ratio to fit especially bar data.")
        , aspectRatioParam("acpect ratio", "Aspect ratio for the diagram.")
        , showMarkersParam("show markers", "When to show markers in line charts.")
        , preparedData(NULL)
        , categories(vislib::Array<vislib::StringA>())
        , hoveredMarker(NULL)
        , hoveredSeries(0)
        , diagram(NULL)
        , selectionCall(NULL)
        , hiddenCall(NULL)
        , markerTextures()
        , preparedSeries()
        , localXIndexToGlobal()
        , xAxis(0.0f)
        , yAxis(0.0f)
        , xTickOff(0.0f)
        , barWidth(0.0f)
        , fontSize(1.0f / 20.0f)
        , legendOffset(0.0f)
        , legendWidth(0.0f)
        , legendHeight(0.0f)
        , legendMargin(0.0f)
        , barWidthRatio(0.8f)
        , selectedSeries(NULL)
        , unselectedColor(vislib::math::Vector<float, 4>(0.5f, 0.5f, 0.5f, 1.0f))
        ,
        // hovering(false),
        hoverPoint()
        , seriesVisible() {

    // segmentation data caller slot
    this->dataCallerSlot.SetCompatibleCall<protein_calls::DiagramCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    this->selectionCallerSlot.SetCompatibleCall<protein_calls::IntSelectionCallDescription>();
    this->MakeSlotAvailable(&this->selectionCallerSlot);

    this->hiddenCallerSlot.SetCompatibleCall<protein_calls::IntSelectionCallDescription>();
    this->MakeSlotAvailable(&this->hiddenCallerSlot);

    param::EnumParam* dt = new param::EnumParam(0);
    dt->SetTypePair(DIAGRAM_TYPE_LINE, "Line");
    dt->SetTypePair(DIAGRAM_TYPE_LINE_STACKED, "Stacked Line");
    dt->SetTypePair(DIAGRAM_TYPE_LINE_STACKED_NORMALIZED, "100% Stacked Line");
    dt->SetTypePair(DIAGRAM_TYPE_COLUMN, "Clustered Column");
    dt->SetTypePair(DIAGRAM_TYPE_COLUMN_STACKED, "Stacked Column");
    dt->SetTypePair(DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED, "100% Stacked Column");
    this->diagramTypeParam.SetParameter(dt);
    this->MakeSlotAvailable(&this->diagramTypeParam);

    param::EnumParam* ds = new param::EnumParam(0);
    ds->SetTypePair(DIAGRAM_STYLE_WIRE, "Wireframe");
    ds->SetTypePair(DIAGRAM_STYLE_FILLED, "Filled");
    this->diagramStyleParam.SetParameter(ds);
    this->MakeSlotAvailable(&this->diagramStyleParam);

    this->numXTicksParam.SetParameter(new param::IntParam(4, 3, 100));
    this->MakeSlotAvailable(&this->numXTicksParam);
    this->numYTicksParam.SetParameter(new param::IntParam(4, 3, 100));
    this->MakeSlotAvailable(&this->numYTicksParam);
    this->drawYLogParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->drawYLogParam);
    this->lineWidthParam.SetParameter(new param::FloatParam(1.0f, 0.1f, 10.0f));
    this->MakeSlotAvailable(&this->lineWidthParam);

    this->foregroundColorParam.SetParameter(new param::StringParam("white"));
    this->fgColor.Set(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->foregroundColorParam);

    param::EnumParam* dc = new param::EnumParam(0);
    dc->SetTypePair(1, "true");
    dc->SetTypePair(0, "false");
    this->drawCategoricalParam.SetParameter(dc);
    this->MakeSlotAvailable(&this->drawCategoricalParam);

    param::EnumParam* sm = new param::EnumParam(0);
    sm->SetTypePair(DIAGRAM_MARKERS_SHOW_ALL, "show all");
    sm->SetTypePair(DIAGRAM_MARKERS_SHOW_SELECTED, "show selected");
    sm->SetTypePair(DIAGRAM_MARKERS_SHOW_NONE, "hide all");
    this->showMarkersParam.SetParameter(sm);
    this->MakeSlotAvailable(&this->showMarkersParam);

    this->showCrosshairParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showCrosshairParam);
    this->showCrosshairToggleParam.SetParameter(
        new param::ButtonParam(core::view::Key::KEY_C, core::view::Modifier::CTRL));
    this->showCrosshairToggleParam.SetUpdateCallback(this, &DiagramRenderer::onCrosshairToggleButton);
    this->MakeSlotAvailable(&this->showCrosshairToggleParam);
    this->showAllParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::CTRL));
    this->showAllParam.SetUpdateCallback(this, &DiagramRenderer::onShowAllButton);
    this->MakeSlotAvailable(&this->showAllParam);
    this->hideAllParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_H, core::view::Modifier::CTRL));
    this->hideAllParam.SetUpdateCallback(this, &DiagramRenderer::onHideAllButton);
    this->MakeSlotAvailable(&this->hideAllParam);

    this->aspectRatioParam.SetParameter(new param::FloatParam(1.0, 0.0));
    this->MakeSlotAvailable(&this->aspectRatioParam);
    this->autoAspectParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->autoAspectParam);
    this->showGuidesParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showGuidesParam);

    seriesVisible.AssertCapacity(100);
    seriesVisible.SetCapacityIncrement(100);
}

/*
 * Diagram2DRenderer::~Diagram2DRenderer (DTOR)
 */
DiagramRenderer::~DiagramRenderer(void) {
    this->Release();
}

/*
 * Diagram2DRenderer::create
 */
bool DiagramRenderer::create() {

    this->LoadIcon("plop.png", protein_calls::DiagramCall::DIAGRAM_MARKER_DISAPPEAR);
    this->LoadIcon("bookmark.png", protein_calls::DiagramCall::DIAGRAM_MARKER_BOOKMARK);
    this->LoadIcon("merge.png", protein_calls::DiagramCall::DIAGRAM_MARKER_MERGE);
    this->LoadIcon("split.png", protein_calls::DiagramCall::DIAGRAM_MARKER_SPLIT);
    this->LoadIcon("exit2.png", protein_calls::DiagramCall::DIAGRAM_MARKER_EXIT);

    return true;
}

/*
 * Diagram2DRenderer::release
 */
void DiagramRenderer::release() {}

bool DiagramRenderer::CalcExtents() {

    // TODO dirty checking and shit

    protein_calls::DiagramCall* diagram = this->dataCallerSlot.CallAs<protein_calls::DiagramCall>();
    if (diagram == NULL)
        return false;
    // execute the call
    if (!(*diagram)(protein_calls::DiagramCall::CallForGetData))
        return false;

    int type = this->diagramTypeParam.Param<param::EnumParam>()->Value();

    // TODO adjust
    bool autoFit = true;
    this->xRange.SetFirst(FLT_MAX);
    this->xRange.SetSecond(-FLT_MAX);
    this->yRange.SetFirst(FLT_MAX);
    this->yRange.SetSecond(-FLT_MAX);
    bool drawCategorical = this->drawCategoricalParam.Param<param::EnumParam>()->Value() != 0;
    if (autoFit) {
        for (int s = 0; s < (int)diagram->GetSeriesCount(); s++) {
            protein_calls::DiagramCall::DiagramSeries* ds = diagram->GetSeries(s);
            const protein_calls::DiagramCall::DiagramMappable* dm = ds->GetMappable();
            if (seriesVisible[s] && isCategoricalMappable(dm) == drawCategorical) {
                vislib::Pair<float, float> xR = dm->GetAbscissaRange(0);
                vislib::Pair<float, float> yR = dm->GetOrdinateRange();
                if (xR.First() < this->xRange.First()) {
                    this->xRange.SetFirst(xR.First());
                }
                if (xR.Second() > this->xRange.Second()) {
                    this->xRange.SetSecond(xR.Second());
                }
                if (yR.First() < this->yRange.First()) {
                    this->yRange.SetFirst(yR.First());
                }
                if (yR.Second() > this->yRange.Second()) {
                    this->yRange.SetSecond(yR.Second());
                }
            }
        }

        switch (type) {
        case DIAGRAM_TYPE_COLUMN_STACKED:
        case DIAGRAM_TYPE_LINE_STACKED:
        case DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED:
        case DIAGRAM_TYPE_LINE_STACKED_NORMALIZED:
            this->yRange.SetFirst(0.0f);
            this->yRange.SetSecond(1.0f);
            break;
        }
    }
    return true;
}

bool DiagramRenderer::GetExtents(mmstd_gl::CallRender2DGL& call) {
    // set the bounding box to 0..1
    call.AccessBoundingBoxes().SetBoundingBox(0.0f - legendOffset - legendWidth, 0.0f - 2.0f * fontSize, 0,
        this->aspectRatioParam.Param<param::FloatParam>()->Value() + fontSize, 1.0f + fontSize, 0);

    // this->CalcExtents();
    ////  ( this->yRange.Second() - this->yRange.First())
    ////            * this->aspectRatioParam.Param<param::FloatParam>()->Value() + this->xRange.First()
    // call.SetBoundingBox(xRange.First(), yRange.First(), xRange.Second(), yRange.Second());
    return true;
}

bool DiagramRenderer::LoadIcon(vislib::StringA filename, int ID) {
    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    void* buf = NULL;
    SIZE_T size = 0;

    // if (pbc.Load(filename)) {
    if ((size = megamol::core::utility::ResourceWrapper::LoadResource(
             this->GetCoreInstance()->Configuration(), filename, &buf)) > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
            for (unsigned int i = 0; i < img.Width() * img.Height(); i++) {
                BYTE r = img.PeekDataAs<BYTE>()[i * 4 + 0];
                BYTE g = img.PeekDataAs<BYTE>()[i * 4 + 1];
                BYTE b = img.PeekDataAs<BYTE>()[i * 4 + 2];
                if (r + g + b > 0) {
                    img.PeekDataAs<BYTE>()[i * 4 + 3] = 255;
                } else {
                    img.PeekDataAs<BYTE>()[i * 4 + 3] = 0;
                }
            }
            markerTextures.Add(vislib::Pair<int, vislib::SmartPtr<vislib_gl::graphics::gl::OpenGLTexture2D>>());
            markerTextures.Last().First() = ID;
            markerTextures.Last().SetSecond(new vislib_gl::graphics::gl::OpenGLTexture2D());
            if (markerTextures.Last().Second()->Create(
                    img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                Log::DefaultLog.WriteError("could not load %s texture.", filename.PeekBuffer());
                ARY_SAFE_DELETE(buf);
                return false;
            }
            markerTextures.Last().Second()->SetFilter(GL_LINEAR, GL_LINEAR);
            ARY_SAFE_DELETE(buf);
            return true;
        } else {
            Log::DefaultLog.WriteError("could not read %s texture.", filename.PeekBuffer());
        }
    } else {
        Log::DefaultLog.WriteError("could not find %s texture.", filename.PeekBuffer());
    }
    return false;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool DiagramRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;

    float aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
    float xObj = x / aspect;

    if (flags & view::MOUSEFLAG_MODKEY_ALT_DOWN) {

        if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
            int type = this->diagramTypeParam.Param<param::EnumParam>()->Value();
            vislib::math::Point<float, 2> mouse(xObj, y);
            vislib::math::Point<float, 2> aspectlessMouse(x, y);
            vislib::math::Point<float, 2> pt3, pt4;
            vislib::math::ShallowPoint<float, 2> pt(mouse.PeekCoordinates());
            vislib::math::ShallowPoint<float, 2> pt2(mouse.PeekCoordinates());
            // printf("checking for hits\n");

            vislib::math::Rectangle<float> legend(
                -legendOffset - legendWidth, 1.0f - legendHeight, -legendOffset, 1.0f);
            if (legend.Contains(aspectlessMouse)) {
                // printf("i am legend\n");
                if (diagram == NULL) {
                    return false;
                }
                float series = 1.0f - legendMargin - mouse.Y();
                series /= theFont.LineHeight(fontSize);
                bool drawCategorical = this->drawCategoricalParam.Param<param::EnumParam>()->Value() != 0;
                vislib::Array<int> visibleSeries;
                visibleSeries.SetCapacityIncrement(10);
                for (int i = 0; i < (int)diagram->GetSeriesCount(); i++) {
                    if (isCategoricalMappable(diagram->GetSeries(i)->GetMappable()) == drawCategorical) {
                        visibleSeries.Add(i);
                    }
                }
                if (series < visibleSeries.Count()) {
                    protein_calls::DiagramCall::DiagramSeries* ds =
                        diagram->GetSeries(visibleSeries[static_cast<int>(series)]);
                    float legendLeft = -legendOffset - legendWidth;
                    if (legendLeft + legendMargin < x && x < legendLeft + legendMargin + 0.6 * fontSize) {
                        // printf("I think I hit the checkbox of series %s\n", ds->GetName());
                        // ds->SetVisible(!ds->GetVisible());
                        seriesVisible[static_cast<int>(series)] = !seriesVisible[static_cast<int>(series)];
                    } else {
                        selectedSeries = ds;
                        // printf("I think I hit series %s\n", selectedSeries->GetName());
                        consumeEvent = true;
                    }
                }
            } else {
                float dist = FLT_MAX;
                int distOffset = -1;
                if (type == DIAGRAM_TYPE_LINE || type == DIAGRAM_TYPE_LINE_STACKED ||
                    type == DIAGRAM_TYPE_LINE_STACKED_NORMALIZED) {

                    for (int i = 0; i < (int)preparedData->Count(); i++) {
                        int leftNeighbor = -1;
                        int rightNeighbor = -1;
                        for (int j = 0; j < (int)(*preparedData)[i]->Count(); j++) {
                            if ((*(*preparedData)[i])[j] != NULL) {
                                if ((*(*preparedData)[i])[j]->GetX() > mouse.GetX()) {
                                    break;
                                }
                                pt.SetPointer((*(*preparedData)[i])[j]->PeekCoordinates());
                                leftNeighbor = j;
                            }
                        }
                        for (int j = static_cast<int>((*preparedData)[i]->Count()) - 1; j > -1; j--) {
                            if ((*(*preparedData)[i])[j] != NULL) {
                                if ((*(*preparedData)[i])[j]->GetX() < mouse.GetX()) {
                                    break;
                                }
                                pt2.SetPointer((*(*preparedData)[i])[j]->PeekCoordinates());
                                rightNeighbor = j;
                            }
                        }
                        if (leftNeighbor == -1 || rightNeighbor == -1 || (rightNeighbor - leftNeighbor) > 1) {
                            continue;
                        }
                        pt3 = pt.Interpolate(pt2, (mouse.GetX() - pt.X()) / (pt2.X() - pt.X()));
                        float d = pt3.Distance(mouse);
                        if (d < dist) {
                            dist = d;
                            distOffset = i;
                        }
                    }
                    if (distOffset != -1 && dist < 0.2f) {
                        // printf("I think I hit series %s[%u]\n", preparedSeries[distOffset]->GetName(), distOffset);
                        selectedSeries = preparedSeries[distOffset];
                        consumeEvent = true;
                    } else {
                        selectedSeries = NULL;
                    }
                } else if (type == DIAGRAM_TYPE_COLUMN || type == DIAGRAM_TYPE_COLUMN_STACKED ||
                           type == DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED) {

                    for (int i = 0; i < (int)preparedData->Count(); i++) {
                        for (int j = 0; j < (int)(*preparedData)[i]->Count(); j++) {
                            if ((*(*preparedData)[i])[j] == NULL) {
                                continue;
                            }
                            float x2, y2;
                            getBarXY(i, j, type, &x2, &y2);
                            float y1 = (*(*preparedData)[i])[j]->GetZ();
                            float xDiff = xObj - x2;
                            if (xDiff > 0.0f && xDiff < barWidth && y > y1 && y < y2) {
                                // printf("I think I hit series %s[%u][%u]\n", preparedSeries[i]->GetName(), i, j);
                                selectedSeries = preparedSeries[i];
                                consumeEvent = true;
                                break;
                            }
                        }
                    }
                    if (!consumeEvent) {
                        selectedSeries = NULL;
                    }
                }
            }
        } else {
        }
    }

    // propagate selection to selection module
    if (selectionCall != NULL) {
        vislib::Array<int> selectedSeriesIndices;
        for (int x = 0; x < (int)this->diagram->GetSeriesCount(); x++) {
            if (this->diagram->GetSeries(x) == this->selectedSeries) {
                selectedSeriesIndices.Add(x);
                break;
            }
        }
        selectionCall->SetSelectionPointer(&selectedSeriesIndices);
        (*selectionCall)(protein_calls::IntSelectionCall::CallForSetSelection);
    }

    // propagate visibility to hidden module
    if (hiddenCall != NULL) {
        vislib::Array<int> hiddenSeriesIndices;
        for (int x = 0; x < (int)this->diagram->GetSeriesCount(); x++) {
            if (!seriesVisible[x]) {
                hiddenSeriesIndices.Add(x);
            }
        }
        hiddenCall->SetSelectionPointer(&hiddenSeriesIndices);
        (*hiddenCall)(protein_calls::IntSelectionCall::CallForSetSelection);
    }

    // hovering
    hoveredMarker = NULL;
    if (preparedData != NULL) {
        for (int s = 0; s < (int)preparedData->Count(); s++) {
            float markerSize = fontSize;
            for (int i = 0; i < (int)preparedSeries[s]->GetMarkerCount(); i++) {
                const protein_calls::DiagramCall::DiagramMarker* m = preparedSeries[s]->GetMarker(i);
                for (int j = 0; j < (int)this->markerTextures.Count(); j++) {
                    if (markerTextures[j].First() == m->GetType()) {
                        markerTextures[j].Second()->Bind();
                        // TODO FIXME BUG WTF does this happen anyway
                        if ((m->GetIndex() > (*preparedData)[s]->Count() - 1) ||
                            (*(*preparedData)[s])[m->GetIndex()] == NULL) {
                            continue;
                        }
                        float mx = (*(*preparedData)[s])[m->GetIndex()]->X();
                        float my = (*(*preparedData)[s])[m->GetIndex()]->Y();
                        mx *= aspect;
                        mx -= markerSize / 2.0f;
                        my -= markerSize / 2.0f;
                        if (mx < x && x < mx + markerSize && my < y && y < my + markerSize) {
                            // printf("hovering over marker %u of series %s\n", i, preparedSeries[s]->GetName());
                            hoveredMarker = m;
                            hoveredSeries = s;
                        }
                    }
                }
            }
        }
    }
    hoverPoint.Set(x, y);

    return consumeEvent;
}


/*
 * DiagramRenderer::onCrosshairToggleButton
 */
bool DiagramRenderer::onCrosshairToggleButton(param::ParamSlot& p) {
    param::BoolParam* bp = this->showCrosshairParam.Param<param::BoolParam>();
    bp->SetValue(!bp->Value());
    return true;
}


/*
 * DiagramRenderer::onShowAllButton
 */
bool DiagramRenderer::onShowAllButton(param::ParamSlot& p) {
    if (this->diagram != NULL) {
        for (int i = 0; i < (int)this->diagram->GetSeriesCount(); i++) {
            // this->diagram->GetSeries(i)->SetVisible(true);
            seriesVisible[i] = true;
        }
    }
    return true;
}


/*
 * DiagramRenderer::onHideAllButton
 */
bool DiagramRenderer::onHideAllButton(param::ParamSlot& p) {
    if (this->diagram != NULL) {
        for (int i = 0; i < (int)this->diagram->GetSeriesCount(); i++) {
            // this->diagram->GetSeries(i)->SetVisible(false);
            seriesVisible[i] = false;
        }
    }
    return true;
}


/*
 * Diagram2DRenderer::Render
 */
bool DiagramRenderer::Render(mmstd_gl::CallRender2DGL& call) {
    // get pointer to Diagram2DCall
    diagram = this->dataCallerSlot.CallAs<protein_calls::DiagramCall>();
    if (diagram == NULL)
        return false;
    // execute the call
    if (!(*diagram)(protein_calls::DiagramCall::CallForGetData))
        return false;

    selectionCall = this->selectionCallerSlot.CallAs<protein_calls::IntSelectionCall>();
    if (selectionCall != NULL) {
        (*selectionCall)(protein_calls::IntSelectionCall::CallForGetSelection);
        if (selectionCall->GetSelectionPointer() != NULL && selectionCall->GetSelectionPointer()->Count() > 0) {
            selectedSeries = diagram->GetSeries((*selectionCall->GetSelectionPointer())[0]);
        } else {
            selectedSeries = NULL;
        }
    }

    while (seriesVisible.Count() < diagram->GetSeriesCount()) {
        seriesVisible.Append(true);
    }

    // do we have visibility information propagated from outside?
    hiddenCall = this->hiddenCallerSlot.CallAs<protein_calls::IntSelectionCall>();
    if (hiddenCall != NULL) {
        (*hiddenCall)(protein_calls::IntSelectionCall::CallForGetSelection);
        if (hiddenCall->GetSelectionPointer() != NULL) {
            for (SIZE_T x = 0; x < seriesVisible.Count(); x++) {
                seriesVisible[x] = true;
            }
            if (hiddenCall->GetSelectionPointer()->Count() > 0) {
                vislib::Array<int>* sel = hiddenCall->GetSelectionPointer();
                for (SIZE_T x = 0; x < sel->Count(); x++) {
                    seriesVisible[(*sel)[x]] = false;
                }
            }
        }
    }

    if (this->foregroundColorParam.IsDirty()) {
        utility::ColourParser::FromString(this->foregroundColorParam.Param<param::StringParam>()->Value().c_str(),
            fgColor.PeekComponents()[0], fgColor.PeekComponents()[1], fgColor.PeekComponents()[2],
            fgColor.PeekComponents()[3]);
    }
    // TODO dirty checking and shit
    this->CalcExtents();

    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // : GL_LINE);
    ::glDisable(GL_LINE_SMOOTH);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_LINE_WIDTH);
    ::glLineWidth(this->lineWidthParam.Param<param::FloatParam>()->Value());

    xAxis = 0.0f;
    yAxis = 0.0f;
    if (yRange.First() < 0.0f && yRange.Second() > 0.0f) {
        xAxis = -yRange.First() / (yRange.Second() - yRange.First());
    }
    if (xRange.First() < 0.0f && xRange.Second() > 0.0f) {
        yAxis = -xRange.First() / (xRange.Second() - xRange.First());
    }

    switch (this->diagramTypeParam.Param<param::EnumParam>()->Value()) {
    case DIAGRAM_TYPE_LINE:
    case DIAGRAM_TYPE_LINE_STACKED:
    case DIAGRAM_TYPE_LINE_STACKED_NORMALIZED:
        drawLineDiagram();
        break;
    case DIAGRAM_TYPE_COLUMN:
    case DIAGRAM_TYPE_COLUMN_STACKED:
    case DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED:
        drawColumnDiagram();
        break;
    }

    float aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
    bool drawLog = this->drawYLogParam.Param<param::BoolParam>()->Value();
    if (this->showCrosshairParam.Param<param::BoolParam>()->Value()) {
        ::glDisable(GL_BLEND);
        ::glDisable(GL_DEPTH_TEST);
        vislib::StringA tmpString;
        float y;
        if (drawLog) {
            y = (float)pow(10, hoverPoint.GetY() * log10(yRange.Second() - yRange.First())) + yRange.First();
        } else {
            y = hoverPoint.GetY() * (yRange.Second() - yRange.First()) + yRange.First();
        }
        tmpString.Format("%f, %f", hoverPoint.GetX() / aspect * (xRange.Second() - xRange.First()) + xRange.First(), y);
        ::glBegin(GL_LINES);
        ::glColor4fv(this->fgColor.PeekComponents());
        ::glVertex3f(hoverPoint.GetX(), 0.0f, decorationDepth);
        ::glVertex3f(hoverPoint.GetX(), 1.0f, decorationDepth);
        ::glVertex3f(0.0f, hoverPoint.GetY(), decorationDepth);
        ::glVertex3f(aspect, hoverPoint.GetY(), decorationDepth);
        ::glEnd();
        theFont.DrawString(hoverPoint.GetX(), hoverPoint.GetY(), fontSize * 0.5f, true, tmpString.PeekBuffer(),
            vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
    }

    if (this->showGuidesParam.Param<param::BoolParam>()->Value()) {
        for (int i = 0; i < (int)diagram->GetGuideCount(); i++) {
            protein_calls::DiagramCall::DiagramGuide* g = diagram->GetGuide(i);
            ::glDisable(GL_BLEND);
            ::glDisable(GL_DEPTH_TEST);
            vislib::StringA tmpString;
            tmpString.Format("%f", g->GetPosition());
            ::glEnable(GL_LINE_STIPPLE);
            ::glLineStipple(12, 0x5555);
            ::glBegin(GL_LINES);
            ::glColor4fv(this->fgColor.PeekComponents());
            float pos;
            switch (g->GetType()) {
            case protein_calls::DiagramCall::DIAGRAM_GUIDE_HORIZONTAL:
                pos = g->GetPosition() - yRange.First();
                pos /= yRange.GetSecond() - yRange.GetFirst();
                ::glVertex3f(0.0f, pos, decorationDepth);
                ::glVertex3f(aspect, pos, decorationDepth);
                ::glEnd();
                ::glDisable(GL_LINE_STIPPLE);
                theFont.DrawString(aspect, pos, fontSize * 0.5f, true, tmpString.PeekBuffer(),
                    vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
                break;
            case protein_calls::DiagramCall::DIAGRAM_GUIDE_VERTICAL:
                pos = g->GetPosition() - xRange.First();
                pos /= xRange.GetSecond() - xRange.GetFirst();
                pos *= aspect;
                ::glVertex3f(pos, 0.0f, decorationDepth);
                ::glVertex3f(pos, 1.0f, decorationDepth);
                ::glEnd();
                ::glDisable(GL_LINE_STIPPLE);
                theFont.DrawString(pos, 1.0f, fontSize * 0.5f, true, tmpString.PeekBuffer(),
                    vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
                break;
            }
        }
    }

    if (hoveredMarker != NULL) {
        float x = (*(*preparedData)[hoveredSeries])[hoveredMarker->GetIndex()]->X();
        float y = (*(*preparedData)[hoveredSeries])[hoveredMarker->GetIndex()]->Y();
        x *= aspect;
        y += fontSize / 2.0f;
        theFont.DrawString(x, y, fontSize * 0.5f, true, hoveredMarker->GetTooltip(),
            vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
        //"w3wt", vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
    }

    return true;
}


void DiagramRenderer::drawYAxis() {
    int numYTicks = 0;
    float aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
    vislib::StringA* yTickText = NULL;
    float* yTicks = NULL;
    if (this->drawYLogParam.Param<param::BoolParam>()->Value()) {
        int startExp, destExp;
        if (yRange.First() == 0.0f) {
            startExp = 0;
        } else {
            startExp = static_cast<int>(ceil(log10(yRange.First())));
        }
        if (yRange.Second() == 0.0f) {
            destExp = 0;
        } else {
            destExp = static_cast<int>(floor(log10(yRange.Second())));
        }
        if (startExp > destExp) {
            destExp = startExp;
        }
        // WARNING: the yRange extremes potentially overlap with [startExp;destExp]
        // making part of this array superfluous. If string drawing is not robust,
        // there might be a detonation
        numYTicks = static_cast<int>(destExp - startExp) + 1 + 2;
        yTickText = new vislib::StringA[numYTicks];
        yTicks = new float[numYTicks];
        yTicks[0] = 0.0f;
        yTickText[0].Format("%.2f", yRange.First());
        yTicks[numYTicks - 1] = 1.0f;
        yTickText[numYTicks - 1].Format("%.2f", yRange.Second());

        for (int i = startExp; i <= destExp; i++) {
            yTickText[i] = vislib::StringA::EMPTY;
            float yVal = (float)pow(10, static_cast<float>(i));
            yTickText[i].Format("%.2f", yVal);
            yTicks[i] = log10(yVal - yRange.First()) / log10(yRange.Second() - yRange.First());
        }
    } else {
        numYTicks = this->numYTicksParam.Param<param::IntParam>()->Value();
        yTickText = new vislib::StringA[numYTicks];
        yTicks = new float[numYTicks];
        float yTickLabel = (yRange.Second() - yRange.First()) / (numYTicks - 1);
        float yTickOff = 1.0f / (numYTicks - 1);
        for (int i = 0; i < numYTicks; i++) {
            yTickText[i] = vislib::StringA::EMPTY;
            yTickText[i].Format("%.2f", yTickLabel * i + yRange.First());
            yTicks[i] = (1.0f / (numYTicks - 1)) * i;
        }
    }

    float arrWidth = 0.025f;
    float arrHeight = 0.012f;
    float tickSize = fontSize;

    ::glBegin(GL_LINES);
    ::glColor4fv(this->fgColor.PeekComponents());
    ::glVertex3f(yAxis, 0.0f, decorationDepth);
    ::glVertex3f(yAxis, 1.0f + 2.0f * arrWidth, decorationDepth);
    ::glVertex3f(yAxis, 1.0f + 2.0f * arrWidth, decorationDepth);
    ::glVertex3f(yAxis - arrHeight, 1.0f + 1.0f * arrWidth, decorationDepth);
    ::glVertex3f(yAxis, 1.0f + 2.0f * arrWidth, decorationDepth);
    ::glVertex3f(yAxis + arrHeight, 1.0f + 1.0f * arrWidth, decorationDepth);
    ::glEnd();

    for (int i = 0; i < numYTicks; i++) {
        ::glBegin(GL_LINES);
        ::glVertex3f(yAxis, yTicks[i], decorationDepth);
        ::glVertex3f(yAxis - tickSize * 0.5f, yTicks[i], decorationDepth);
        ::glEnd();
        theFont.DrawString(yAxis - tickSize * 0.5f, yTicks[i], fontSize, true, yTickText[i],
            vislib::graphics::AbstractFont::ALIGN_RIGHT_TOP);
    }
    delete[] yTickText;
    delete[] yTicks;
}


void DiagramRenderer::drawXAxis(XAxisTypes xType) {
    if (diagram->GetSeriesCount() == 0) {
        xTickOff = 0.0f;
    }
    int numXTicks;
    switch (xType) {
    case DIAGRAM_XAXIS_FLOAT:
        numXTicks = this->numXTicksParam.Param<param::IntParam>()->Value();
        break;
    case DIAGRAM_XAXIS_INTEGRAL: {
        // numXTicks = 0;
        // for (int i = 0; i < diagram->GetSeriesCount(); i++) {
        //    DiagramCall::DiagramSeries *ds = diagram->GetSeries(i);
        //    const DiagramCall::DiagramMappable *dm = ds->GetMappable();
        //    if (dm->GetDataCount() > numXTicks) {
        //        numXTicks = dm->GetDataCount();
        //    }
        //}
        // numXTicks++;
        numXTicks = (int)xValues.Count();
    } break;
    case DIAGRAM_XAXIS_CATEGORICAL:
        numXTicks = static_cast<int>(categories.Count() + 1);
        break;
    }
    vislib::StringA* xTickText = new vislib::StringA[numXTicks];
    float xTickLabel = (xRange.Second() - xRange.First()) / (numXTicks - 1);
    for (int i = 0; i < numXTicks; i++) {
        xTickText[i] = vislib::StringA::EMPTY;
        switch (xType) {
        case DIAGRAM_XAXIS_FLOAT:
            xTickText[i].Format("%.2f", xTickLabel * i + xRange.First());
            break;
        case DIAGRAM_XAXIS_INTEGRAL:
            xTickText[i].Format("%u", i);
            break;
        case DIAGRAM_XAXIS_CATEGORICAL:
            // not needed
            break;
        }
    }
    float aspect;
    if (this->autoAspectParam.Param<param::BoolParam>()->Value()) {
        switch (xType) {
        case DIAGRAM_XAXIS_FLOAT:
            // cannot think of anything better actually
            aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
            break;
        case DIAGRAM_XAXIS_INTEGRAL:
            aspect = numXTicks * theFont.LineWidth(fontSize, xTickText[numXTicks - 1].PeekBuffer()) * 1.5f;
            break;
        case DIAGRAM_XAXIS_CATEGORICAL: {
            float wMax = 0.0f;
            for (int i = 0; i < (int)categories.Count(); i++) {
                float w = theFont.LineWidth(fontSize, categories[i].PeekBuffer());
                if (w > wMax) {
                    wMax = w;
                }
            }
            wMax *= 2.0f;
            aspect = wMax * categories.Count();
        } break;
        }
        this->aspectRatioParam.Param<param::FloatParam>()->SetValue(aspect);
    } else {
        aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
    }
    float arrWidth = 0.025f / aspect;
    float arrHeight = 0.012f;
    float tickSize = fontSize;

    ::glPushMatrix();
    ::glScalef(aspect, 1.0f, 1.0f);
    ::glBegin(GL_LINES);
    ::glColor4fv(this->fgColor.PeekComponents());
    ::glVertex3f(0.0f, xAxis, decorationDepth);
    ::glVertex3f(1.0f + 2.0f * arrWidth, xAxis, decorationDepth);
    ::glVertex3f(1.0f + 2.0f * arrWidth, xAxis, decorationDepth);
    ::glVertex3f(1.0f + 1.0f * arrWidth, xAxis - arrHeight, decorationDepth);
    ::glVertex3f(1.0f + 2.0f * arrWidth, xAxis, decorationDepth);
    ::glVertex3f(1.0f + 1.0f * arrWidth, xAxis + arrHeight, decorationDepth);
    ::glEnd();
    xTickOff = 1.0f / (numXTicks - 1);
    for (int i = 0; i < numXTicks; i++) {
        ::glBegin(GL_LINES);
        ::glVertex3f(xTickOff * i, xAxis, decorationDepth);
        ::glVertex3f(xTickOff * i, xAxis - tickSize * 0.5f, decorationDepth);
        ::glEnd();
    }
    ::glPopMatrix();

    switch (xType) {
    case DIAGRAM_XAXIS_CATEGORICAL: {
        for (int i = 0; i < numXTicks - 1; i++) {
            theFont.DrawString(aspect * (xTickOff * (i + 0.5f)), xAxis - tickSize * 0.5f, fontSize, true,
                categories[i].PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
        }
    } break;
    case DIAGRAM_XAXIS_INTEGRAL: {
        float needed = theFont.LineWidth(fontSize, xTickText[numXTicks - 1]);
        int step = vislib::math::Max(static_cast<int>(needed / (xTickOff / aspect)), 1);
        for (int i = 0; i < numXTicks - 1; i += step) {
            theFont.DrawString(aspect * (xTickOff * (i + 0.5f)), xAxis - tickSize * 0.5f, fontSize, true, xTickText[i],
                vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
        }
    } break;
    case DIAGRAM_XAXIS_FLOAT: {
        for (int i = 0; i < numXTicks; i++) {
            theFont.DrawString(aspect * (xTickOff * i), xAxis - tickSize * 0.5f, fontSize, true, xTickText[i],
                vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
        }
    } break;
    }

    delete[] xTickText;
}

void DiagramRenderer::drawLegend() {
    legendWidth = 0.0f;
    vislib::StringA s;
    s.Format("%.2f", yRange.Second());
    legendOffset = theFont.LineWidth(fontSize, s) + fontSize; // 3.0f * fontSize;
    bool drawCategorical = this->drawCategoricalParam.Param<param::EnumParam>()->Value() != 0;
    int cnt = 0;
    for (int s = 0; s < (int)diagram->GetSeriesCount(); s++) {
        protein_calls::DiagramCall::DiagramSeries* ds = diagram->GetSeries(s);
        if (isCategoricalMappable(ds->GetMappable()) == drawCategorical) {
            float w = theFont.LineWidth(fontSize, ds->GetName());
            if (w > legendWidth) {
                legendWidth = w;
            }
            cnt++;
        }
    }
    legendMargin = legendWidth * 0.1f;
    legendHeight = theFont.LineHeight(fontSize) * cnt + 2.0f * legendMargin;
    legendWidth = legendWidth + 2.0f * legendMargin + fontSize * 0.8f;
    float legendLeft = -legendOffset - legendWidth;
    ::glBegin(GL_LINE_STRIP);
    ::glColor4fv(this->fgColor.PeekComponents());
    ::glVertex3f(-legendOffset, 1.0f, decorationDepth);
    ::glVertex3f(legendLeft, 1.0f, decorationDepth);
    ::glVertex3f(legendLeft, 1.0f - legendHeight, decorationDepth);
    ::glVertex3f(-legendOffset, 1.0f - legendHeight, decorationDepth);
    ::glVertex3f(-legendOffset, 1.0f, decorationDepth);
    ::glEnd();
    cnt = 0;
    for (int s = 0; s < (int)diagram->GetSeriesCount(); s++) {
        protein_calls::DiagramCall::DiagramSeries* ds = diagram->GetSeries(s);
        if (isCategoricalMappable(ds->GetMappable()) == drawCategorical) {
            if (selectedSeries == NULL || *selectedSeries == *ds) {
                ::glColor4fv(ds->GetColor().PeekComponents());
            } else {
                ::glColor4fv(unselectedColor.PeekComponents());
            }
            float y = 1.0f - legendMargin - theFont.LineHeight(fontSize) * cnt;
            theFont.DrawString(-legendOffset - legendMargin, y, fontSize, true, ds->GetName(),
                vislib::graphics::AbstractFont::ALIGN_RIGHT_TOP);
            ::glBegin(GL_LINE_STRIP);
            ::glVertex3f(legendLeft + legendMargin, y - 0.2f * fontSize, decorationDepth);
            ::glVertex3f(legendLeft + legendMargin, y - 0.8f * fontSize, decorationDepth);
            ::glVertex3f(legendLeft + legendMargin + 0.6f * fontSize, y - 0.8f * fontSize, decorationDepth);
            ::glVertex3f(legendLeft + legendMargin + 0.6f * fontSize, y - 0.2f * fontSize, decorationDepth);
            ::glVertex3f(legendLeft + legendMargin, y - 0.2f * fontSize, decorationDepth);
            ::glEnd();
            if (seriesVisible[s]) {
                ::glBegin(GL_LINES);
                ::glVertex3f(legendLeft + legendMargin, y - 0.2f * fontSize, decorationDepth);
                ::glVertex3f(legendLeft + legendMargin + 0.6f * fontSize, y - 0.8f * fontSize, decorationDepth);
                ::glVertex3f(legendLeft + legendMargin + 0.6f * fontSize, y - 0.2f * fontSize, decorationDepth);
                ::glVertex3f(legendLeft + legendMargin, y - 0.8f * fontSize, decorationDepth);
                ::glEnd();
            }
            cnt++;
        }
    }
}

void DiagramRenderer::getBarXY(int series, int index, int type, float* x, float* y) {
    if (isCategoricalMappable(preparedSeries[series]->GetMappable())) {
        *x = (*(*preparedData)[series])[index]->GetX();
    } else {
        *x = static_cast<float>(index);
    }
    if (type == DIAGRAM_TYPE_COLUMN_STACKED || type == DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED) {
        *x = yAxis + (*x + 0.5f) * xTickOff - xTickOff * 0.5f * barWidthRatio;
    } else {
        *x = yAxis + (*x + 0.5f) * xTickOff - xTickOff * 0.5f * barWidthRatio + series * barWidth;
    }
    *y = (*(*preparedData)[series])[index]->GetY();
}

int floatComp(const float& lhs, const float& rhs) {
    if (lhs < rhs) {
        return -1;
    } else if (lhs > rhs) {
        return 1;
    } else {
        return 0;
    }
}

void DiagramRenderer::prepareData(bool stack, bool normalize, bool drawCategorical) {
    if (preparedData != NULL) {
        // TODO: why??
        // preparedData->Clear();
        delete preparedData;
        preparedData = NULL;
    }
    preparedData = new vislib::PtrArray<vislib::PtrArray<vislib::math::Point<float, 3>>>();
    vislib::Array<float> maxYValues;
    xValues.Clear();
    xValues.SetCapacityIncrement(10);
    bool drawLog = this->drawYLogParam.Param<param::BoolParam>()->Value();
    categories.Clear();
    preparedSeries.Clear();
    vislib::StringA tmpString;
    int maxCount = 0;
    float maxStackedY = -FLT_MAX;
    float x, y, z, tempX;
    // find "broadest" series as well as all distinct abscissa values (for stacking)
    for (int s = 0; s < (int)diagram->GetSeriesCount(); s++) {
        protein_calls::DiagramCall::DiagramSeries* ds = diagram->GetSeries(s);
        const protein_calls::DiagramCall::DiagramMappable* dm = ds->GetMappable();
        if (dm->GetDataCount() > maxCount) {
            maxCount = dm->GetDataCount();
        }
        if (!seriesVisible[s] || isCategoricalMappable(dm) != drawCategorical) {
            continue;
        }
        for (int i = 0; i < dm->GetDataCount(); i++) {
            if (drawCategorical) {
                bool ret = dm->GetAbscissaValue(i, 0, &tmpString);
                if (ret) {
                    int idx = static_cast<int>(categories.IndexOf(tmpString));
                    if (idx == vislib::Array<vislib::StringA>::INVALID_POS) {
                        categories.Add(tmpString);
                        // idx = static_cast<int>(categories.Count() - 1);
                        xValues.Add(static_cast<float>(idx));
                    }
                }
            } else {
                bool ret = dm->GetAbscissaValue(i, 0, &x);
                if (ret) {
                    if (!xValues.Contains(x)) {
                        xValues.Add(x);
                    }
                }
            }
        }
    }
    xValues.Sort(&floatComp);
    maxYValues.SetCount(xValues.Count());
    for (int i = 0; i < (int)maxYValues.Count(); i++) {
        maxYValues[i] = 0.0f;
    }
    // there is a difference between not finding an x value and having a hole which is explicitly returned as NULL
    localXIndexToGlobal.SetCount(diagram->GetSeriesCount());

#if 1
    int cntSeries = 0;
    for (int s = 0; s < (int)diagram->GetSeriesCount(); s++) {
        protein_calls::DiagramCall::DiagramSeries* ds = diagram->GetSeries(s);
        const protein_calls::DiagramCall::DiagramMappable* dm = ds->GetMappable();
        if (!seriesVisible[s] || isCategoricalMappable(dm) != drawCategorical) {
            continue;
        }
        cntSeries++;
        localXIndexToGlobal[cntSeries - 1].SetCount(dm->GetDataCount());
        if ((int)preparedData->Count() < cntSeries) {
            preparedData->Append(new vislib::PtrArray<vislib::math::Point<float, 3>>());
            preparedSeries.Append(ds);
            (*preparedData)[preparedData->Count() - 1]->SetCount(xValues.Count());
        }
        int globalX = 0;
        bool inHole = true, ret;
        for (int localX = 0; localX < dm->GetDataCount(); localX++) {
            if (drawCategorical) {
                ret = dm->GetAbscissaValue(localX, 0, &tmpString);
                if (ret) {
                    int idx = static_cast<int>(categories.IndexOf(tmpString));
                    tempX = static_cast<float>(idx);
                    while (xValues[globalX] < tempX) {
                        if (inHole) {
                            (*(*preparedData)[cntSeries - 1])[globalX] = NULL;
                        } else {
                            (*(*preparedData)[cntSeries - 1])[globalX] = new vislib::math::Point<float, 3>(x, y, z);
                        }
                        globalX++;
                    }
                    ASSERT(xValues[globalX] == tempX);
                    localXIndexToGlobal[cntSeries - 1][localX] = globalX;
                    y = dm->GetOrdinateValue(localX);
                }
            } else {
                ret = dm->GetAbscissaValue(localX, 0, &tempX);
                if (ret) {
                    while (xValues[globalX] < tempX) {
                        if (inHole) {
                            (*(*preparedData)[cntSeries - 1])[globalX] = NULL;
                        } else {
                            (*(*preparedData)[cntSeries - 1])[globalX] = new vislib::math::Point<float, 3>(x, y, z);
                        }
                        globalX++;
                    }
                    ASSERT(xValues[globalX] == tempX);
                    localXIndexToGlobal[cntSeries - 1][localX] = globalX;
                    x = tempX - xRange.First();
                    x /= xRange.Second() - xRange.First();
                    y = dm->GetOrdinateValue(localX);
                }
            }
            if (ret) {
                z = 0.0f;
                (*(*preparedData)[cntSeries - 1])[globalX] = new vislib::math::Point<float, 3>(x, y, z);
            }
            inHole = !ret;
        }
    }
#else // old, wrong implementation
    for (int i = 0; i < xValues.Count(); i++) {
        int cntSeries = 0;
        for (int s = 0; s < diagram->GetSeriesCount(); s++) {
            DiagramCall::DiagramSeries* ds = diagram->GetSeries(s);
            const DiagramCall::DiagramMappable* dm = ds->GetMappable();
            if (!seriesVisible[s] || isCategoricalMappable(dm) != drawCategorical) {
                continue;
            }
            cntSeries++;
            localXIndexToGlobal[cntSeries - 1].SetCount(dm->GetDataCount());
            if (preparedData->Count() < cntSeries) {
                preparedData->Append(new vislib::PtrArray<vislib::math::Point<float, 3>>());
                preparedSeries.Append(ds);
                (*preparedData)[preparedData->Count() - 1]->SetCount(xValues.Count());
            }
            bool found = false;
            for (int j = 0; j < dm->GetDataCount(); j++) {
                bool ret;
                if (drawCategorical) {
                    ret = dm->GetAbscissaValue(j, 0, &tmpString);
                    if (ret) {
                        int idx = static_cast<int>(categories.IndexOf(tmpString));
                        x = static_cast<float>(idx);
                        if (idx == i) {
                            localXIndexToGlobal[cntSeries - 1][j] = xValues.IndexOf(x);
                            y = dm->GetOrdinateValue(j);
                            found = true;
                        }
                    } else {
                        // this is a hole! but where?
                    }
                } else {
                    ret = dm->GetAbscissaValue(j, 0, &x);
                    if (ret) {
                        if (x == xValues[i]) {
                            localXIndexToGlobal[cntSeries - 1][j] = i;
                            x -= xRange.First();
                            x /= xRange.Second() - xRange.First();
                            y = dm->GetOrdinateValue(j);
                            found = true;
                        }
                    } else {
                        // this is a hole! but where?
                    }
                }
                if (found) {
                    z = 0.0f;
                    // if (y < 1.0f) {
                    //    printf("Michael luegt und serie %s hat (%f, %f)\n", preparedSeries[cntSeries - 1]->GetName(),
                    //    x, y);
                    //}
                    (*(*preparedData)[cntSeries - 1])[i] = new vislib::math::Point<float, 3>(x, y, z);
                    break;
                }
            }
            if (!found) {
                (*(*preparedData)[cntSeries - 1])[i] = NULL;
            }
        }
    }
#endif
    // for (int s = 0; s < preparedData->Count(); s++) {
    //    printf("series %u:", s);
    //    for (int i = 0; i < xValues.Count(); i++) {
    //        if ((*(*preparedData)[s])[i] != NULL) {
    //            printf("(%f,%f,%f),", (*(*preparedData)[s])[i]->GetX(), (*(*preparedData)[s])[i]->GetY(),
    //            (*(*preparedData)[s])[i]->GetZ());
    //        } else {
    //            printf("(NULL),");
    //        }
    //    }
    //    printf("\n");
    //}

    // now we could directly stack and normalize
    if (stack) {
        for (int i = 0; i < (int)xValues.Count(); i++) {
            float sum = 0.0f;
            for (int s = 0; s < (int)preparedData->Count(); s++) {
                if ((*(*preparedData)[s])[i] != NULL) {
                    float y = (*(*preparedData)[s])[i]->GetY();
                    (*(*preparedData)[s])[i]->SetZ(sum);
                    sum += y;
                    (*(*preparedData)[s])[i]->SetY(sum);
                }
            }
            maxYValues[i] = sum;
            if (sum > maxStackedY) {
                maxStackedY = sum;
            }
        }
    }
    float norm = yRange.Second() - yRange.First();
    norm = drawLog ? log10(norm) : norm;
    for (int i = 0; i < (int)xValues.Count(); i++) {
        for (int s = 0; s < (int)preparedData->Count(); s++) {
            if ((*(*preparedData)[s])[i] != NULL) {
                float y = (*(*preparedData)[s])[i]->GetY();
                float z = (*(*preparedData)[s])[i]->GetZ();
                if (stack) {
                    if (normalize) {
                        y /= maxYValues[i];
                        z /= maxYValues[i];
                    } else {
                        y /= maxStackedY;
                        z /= maxStackedY;
                    }
                } else {
                    if (drawLog) {
                        y = log10(y - yRange.First()) / norm;
                        z = log10(z - yRange.First()) / norm;
                    } else {
                        y = (y - yRange.First()) / norm;
                        z = (z - yRange.First()) / norm;
                    }
                }
                (*(*preparedData)[s])[i]->SetZ(z);
                (*(*preparedData)[s])[i]->SetY(y);
            }
        }
    }
    if (!normalize && stack) {
        this->yRange.SetSecond(maxStackedY * (yRange.Second() - yRange.First()) + yRange.First());
    }
}

// EVIL EVIL HACK HACK
void DiagramRenderer::dump() {
    vislib::sys::BufferedFile bf;
    bf.Open("dumm.stat", vislib::sys::BufferedFile::WRITE_ONLY, vislib::sys::BufferedFile::SHARE_READ,
        vislib::sys::BufferedFile::CREATE_OVERWRITE);
    for (int i = 0; i < (int)(*preparedData)[0]->Count(); i++) {
        vislib::sys::WriteFormattedLineToFile(bf, "## Frame %u\n", i);
        for (int s = 0; s < (int)preparedData->Count(); s++) {
            if ((*(*preparedData)[s])[i] != NULL) {
                vislib::sys::WriteFormattedLineToFile(bf, "#C %u %u\n", s + 1,
                    static_cast<int>(vislib::math::Min(
                        vislib::math::Max((*(*preparedData)[s])[i]->GetY() * 23000.0f / 70.0f, 3.0f), 20.0f)));
            }
        }
    }
    for (int s = 0; s < (int)preparedData->Count(); s++) {
        for (int i = 0; i < (int)preparedSeries[s]->GetMarkerCount(); i++) {
            // WARNING s is synchronized to global series counter since no series that cannot be drawn are added for
            // proteins For the rest of the universe THIS IS WRONG
            const protein_calls::DiagramCall::DiagramMarker* m = preparedSeries[s]->GetMarker(i);
            if (m->GetType() == protein_calls::DiagramCall::DIAGRAM_MARKER_MERGE && m->GetUserData() != NULL) {
                vislib::Array<int>* partners = reinterpret_cast<vislib::Array<int>*>(m->GetUserData());
                for (int p = 0; p < (int)partners->Count(); p++) {
                    int idx = localXIndexToGlobal[s][m->GetIndex()];
                    vislib::sys::WriteFormattedLineToFile(
                        bf, "#F %u[%u]=>%u[%u] %u\n", (*partners)[p] + 1, idx - 1, s + 1, idx, 3);
                }
            } else if (m->GetType() == protein_calls::DiagramCall::DIAGRAM_MARKER_SPLIT && m->GetUserData() != NULL) {
                vislib::Array<int>* partners = reinterpret_cast<vislib::Array<int>*>(m->GetUserData());
                for (int p = 0; p < (int)partners->Count(); p++) {
                    // Log::DefaultLog.WriteInfo( "#F %u[%u]=>%u[%u] %u", s + 1, m->GetIndex(),
                    // (*partners)[p] + 1, m->GetIndex() + 1, 3);
                    int idx = localXIndexToGlobal[s][m->GetIndex()];
                    vislib::sys::WriteFormattedLineToFile(
                        bf, "#F %u[%u]=>%u[%u] %u\n", (*partners)[p] + 1, idx - 1, s + 1, idx, 3);
                }
            }
        }
    }
    vislib::sys::WriteFormattedLineToFile(bf, "## MaxClustID %u\n", preparedData->Count());
    bf.Close();
}

void DiagramRenderer::drawLineDiagram() {

    bool drawCategorical = this->drawCategoricalParam.Param<param::EnumParam>()->Value() != 0;
    if (drawCategorical) {
        return;
    }
    float aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
    glDisable(GL_BLEND);
    this->drawXAxis(DIAGRAM_XAXIS_FLOAT);
    int type = this->diagramTypeParam.Param<param::EnumParam>()->Value();
    switch (type) {
    case DIAGRAM_TYPE_LINE:
        prepareData(false, false, drawCategorical);
        break;
    case DIAGRAM_TYPE_LINE_STACKED:
        prepareData(true, false, drawCategorical);
        break;
    case DIAGRAM_TYPE_LINE_STACKED_NORMALIZED:
        prepareData(true, true, drawCategorical);
        break;
    }
    this->drawYAxis();
    this->drawLegend();

    // HACK HACK
    bool d = false;
    if (d) {
        this->dump();
    }

    ::glBlendFunc(GL_ONE, GL_ONE);
    ::glDisable(GL_DEPTH_TEST);
    GLenum drawMode = 0;
    if (this->diagramStyleParam.Param<param::EnumParam>()->Value() == DIAGRAM_STYLE_FILLED) {
        drawMode = GL_TRIANGLE_STRIP;
        ::glEnable(GL_BLEND);
    } else {
        drawMode = GL_LINE_STRIP;
        ::glDisable(GL_BLEND);
    }
    for (int s = 0; s < (int)preparedData->Count(); s++) {
        if ((*preparedData)[s]->Count() < 2) {
            continue;
        }
        ::glBegin(drawMode);
        if (selectedSeries == NULL || *selectedSeries == *preparedSeries[s]) {
            ::glColor4fv(preparedSeries[s]->GetColor().PeekComponents());
        } else {
            ::glColor4fv(unselectedColor.PeekComponents());
        }
        for (int i = 0; i < (int)(*preparedData)[s]->Count(); i++) {
            if ((*(*preparedData)[s])[i] != NULL) {
                ::glVertex2f((*(*preparedData)[s])[i]->GetX() * aspect, (*(*preparedData)[s])[i]->GetY());
                if (drawMode == GL_TRIANGLE_STRIP) {
                    ::glVertex2f((*(*preparedData)[s])[i]->GetX() * aspect, (*(*preparedData)[s])[i]->GetZ());
                }
            } else {
                ::glEnd();
                ::glBegin(drawMode);
            }
        }
        ::glEnd();
    }
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_TEXTURE);
    ::glEnable(GL_TEXTURE_2D);
    //::glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    ::glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    int showMarkers = this->showMarkersParam.Param<param::EnumParam>()->Value();
    if (showMarkers != DIAGRAM_MARKERS_SHOW_NONE) {
        for (int s = 0; s < (int)preparedData->Count(); s++) {
            if (showMarkers == DIAGRAM_MARKERS_SHOW_ALL || preparedSeries[s] == selectedSeries) {
                float markerSize = fontSize;
                for (int i = 0; i < (int)preparedSeries[s]->GetMarkerCount(); i++) {
                    const protein_calls::DiagramCall::DiagramMarker* m = preparedSeries[s]->GetMarker(i);
                    for (int j = 0; j < (int)this->markerTextures.Count(); j++) {
                        if (markerTextures[j].First() == m->GetType()) {
                            int idx = localXIndexToGlobal[s][m->GetIndex()];
                            if ((*(*preparedData)[s])[idx] == NULL) {
                                continue;
                            }
                            markerTextures[j].Second()->Bind();
                            float x = (*(*preparedData)[s])[idx]->X();
                            float y = (*(*preparedData)[s])[idx]->Y();
                            x *= aspect;
                            x -= markerSize / 2.0f;
                            y -= markerSize / 2.0f;
                            ::glBegin(GL_TRIANGLE_STRIP);
                            if (selectedSeries == NULL || *selectedSeries == *preparedSeries[s]) {
                                ::glColor4fv(preparedSeries[s]->GetColor().PeekComponents());
                            } else {
                                ::glColor4fv(unselectedColor.PeekComponents());
                            }
                            ::glTexCoord2f(0.0f, 1.0f);
                            ::glVertex3f(x, y, decorationDepth - 0.5f);
                            ::glTexCoord2f(0.0f, 0.0f);
                            ::glVertex3f(x, y + markerSize, decorationDepth - 0.5f);
                            ::glTexCoord2f(1.0f, 1.0f);
                            ::glVertex3f(x + markerSize, y, decorationDepth - 0.5f);
                            ::glTexCoord2f(1.0f, 0.0f);
                            ::glVertex3f(x + markerSize, y + markerSize, decorationDepth - 0.5f);
                            ::glEnd();
                            continue;
                        }
                    }
                }
            }
        }
    }
    ::glDisable(GL_TEXTURE);
    ::glDisable(GL_TEXTURE_2D);
    ::glBindTexture(GL_TEXTURE_2D, 0);
}

void DiagramRenderer::drawColumnDiagram() {
    barWidth = 0.0f;

    glDisable(GL_BLEND);
    float aspect = this->aspectRatioParam.Param<param::FloatParam>()->Value();
    bool drawCategorical = this->drawCategoricalParam.Param<param::EnumParam>()->Value() != 0;
    int type = this->diagramTypeParam.Param<param::EnumParam>()->Value();

    switch (type) {
    case DIAGRAM_TYPE_COLUMN:
        prepareData(false, false, drawCategorical);
        break;
    case DIAGRAM_TYPE_COLUMN_STACKED:
        prepareData(true, false, drawCategorical);
        break;
    case DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED:
        prepareData(true, true, drawCategorical);
        break;
    }
    this->drawYAxis();
    this->drawLegend();

    vislib::StringA tmpString;

    if (drawCategorical == false) {
        this->drawXAxis(DIAGRAM_XAXIS_INTEGRAL);
        barWidth = (xTickOff * barWidthRatio) / preparedData->Count();
    } else {
        this->drawXAxis(DIAGRAM_XAXIS_CATEGORICAL);
        barWidth = (xTickOff * barWidthRatio) / preparedData->Count();
    }
    if (type == DIAGRAM_TYPE_COLUMN_STACKED || type == DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED) {
        barWidth = xTickOff * barWidthRatio;
    }

    ::glPushMatrix();
    ::glScalef(aspect, 1.0f, 1.0f);
    GLenum drawMode = 0;
    if (this->diagramStyleParam.Param<param::EnumParam>()->Value() == DIAGRAM_STYLE_FILLED) {
        drawMode = GL_TRIANGLE_STRIP;
    } else {
        drawMode = GL_LINE_STRIP;
    }
    for (int s = 0; s < (int)preparedData->Count(); s++) {
        float x, y, y1;
        for (int i = 0; i < (int)(*preparedData)[s]->Count(); i++) {
            if ((*(*preparedData)[s])[i] == NULL) {
                continue;
            }
            getBarXY(s, i, type, &x, &y);
            y1 = (*(*preparedData)[s])[i]->GetZ();
            ::glBegin(drawMode);
            if (selectedSeries == NULL || *selectedSeries == *preparedSeries[s]) {
                ::glColor4fv(preparedSeries[s]->GetColor().PeekComponents());
            } else {
                ::glColor4fv(unselectedColor.PeekComponents());
            }
            if (drawMode == GL_TRIANGLE_STRIP) {
                ::glVertex2f(x + barWidth * 0.1f, y);
                ::glVertex2f(x + barWidth * 0.1f, y1);
                ::glVertex2f(x + barWidth * 0.9f, y);
                ::glVertex2f(x + barWidth * 0.9f, y1);
                ::glEnd();
            } else {
                ::glVertex2f(x + barWidth * 0.1f, y);
                ::glVertex2f(x + barWidth * 0.1f, y1);
                ::glVertex2f(x + barWidth * 0.9f, y1);
                ::glVertex2f(x + barWidth * 0.9f, y);
                ::glVertex2f(x + barWidth * 0.1f, y);
                ::glEnd();
            }
        }
    }
    ::glPopMatrix();
}
