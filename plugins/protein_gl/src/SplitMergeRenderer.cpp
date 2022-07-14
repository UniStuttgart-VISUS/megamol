#include "SplitMergeRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/math/Rectangle.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"
#include <math.h>
//#include "mmcore/misc/ImageViewer.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "vislib/math/FastMap.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include <float.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_gl;
using namespace vislib_gl::graphics::gl;
using megamol::core::utility::log::Log;

vislib::Array<int>* SplitMergeRenderer::FastMapWrapper::sortedSeries;
protein_calls::SplitMergeCall* SplitMergeRenderer::FastMapWrapper::diagram;

/*
 * SplitMergeRenderer::SplitMergeRenderer (CTOR)
 */
SplitMergeRenderer::SplitMergeRenderer(void)
        : mmstd_gl::Renderer2DModuleGL()
        , dataCallerSlot("getData", "Connects the diagram rendering with data storage.")
        , selectionCallerSlot("getSelection", "Connects the diagram rendering with selection storage.")
        , hiddenCallerSlot("getHidden", "Connects the diagram rendering with visibility storage.")
        , foregroundColorParam("foregroundCol", "The color of the diagram decorations.")
        , showGuidesParam("show guides", "Show defined guides in the diagram.")
        , visibilityFromSelection("sel->vis", "infer visibility from selection")
        , numVisibilityPropagationRounds(
              "propagation rounds", "how many times visibility is propagated through transitions")
        , bounds()
        , selectionLevel()
        , fgColor(vislib::math::Vector<float, 4>(1.0f, 1.0f, 1.0f, 1.0f))
        , diagram(NULL)
        , selectedSeries(NULL)
        , selectionCall(NULL)
        , hiddenCall(NULL)
        , unselectedColor(vislib::math::Vector<float, 4>(0.5f, 0.5f, 0.5f, 1.0f))
        , noseLength(0.1f)
        , fontSize(1.0f)
        , seriesSpacing(2.0f)
        , sortedSeries()
        , seriesVisible()
// sortedSeriesInverse(),
{

    // segmentation data caller slot
    this->dataCallerSlot.SetCompatibleCall<protein_calls::SplitMergeCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    this->selectionCallerSlot.SetCompatibleCall<protein_calls::IntSelectionCallDescription>();
    this->MakeSlotAvailable(&this->selectionCallerSlot);

    this->hiddenCallerSlot.SetCompatibleCall<protein_calls::IntSelectionCallDescription>();
    this->MakeSlotAvailable(&this->hiddenCallerSlot);

    this->foregroundColorParam.SetParameter(new param::StringParam("white"));
    this->fgColor.Set(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->foregroundColorParam);
    this->showGuidesParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showGuidesParam);
    this->visibilityFromSelection.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->visibilityFromSelection);
    this->numVisibilityPropagationRounds.SetParameter(new param::IntParam(1, 1, 50));
    this->MakeSlotAvailable(&this->numVisibilityPropagationRounds);
    this->seriesVisible.AssertCapacity(100);
    this->seriesVisible.SetCount(100);
}

/*
 * SplitMergeRenderer::~SplitMergeRenderer (DTOR)
 */
SplitMergeRenderer::~SplitMergeRenderer(void) {
    this->Release();
}

/*
 * SplitMergeRenderer::create
 */
bool SplitMergeRenderer::create() {

    this->numChars = 256; // 0x110000;
    this->fontBase = glGenPathsNV(numChars);
    // this->pathBase = glGenPathsNV(1000);
    GLuint templatePathObject = ~0;
    /*
            glPathCommandsNV(templatePathObject, 0, NULL, 0, GL_FLOAT, NULL);
            glPathParameterfNV(templatePathObject, GL_PATH_STROKE_WIDTH_NV, emScale*0.1f);
            glPathParameteriNV(templatePathObject, GL_PATH_JOIN_STYLE_NV, GL_MITER_TRUNCATE_NV);
            glPathParameterfNV(templatePathObject, GL_PATH_MITER_LIMIT_NV, 1.0);*/

    glPathGlyphRangeNV(fontBase, GL_SYSTEM_FONT_NAME_NV, "Verdana", GL_NONE, 0, numChars, GL_SKIP_MISSING_GLYPH_NV,
        templatePathObject, this->fontSize);
    glPathGlyphRangeNV(fontBase, GL_SYSTEM_FONT_NAME_NV, "Arial", GL_BOLD_BIT_NV, 0, numChars, GL_SKIP_MISSING_GLYPH_NV,
        templatePathObject, this->fontSize);
    glPathGlyphRangeNV(fontBase, GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV, 0, numChars, GL_USE_MISSING_GLYPH_NV,
        templatePathObject, this->fontSize);

    return true;
}

/*
 * SplitMergeRenderer::release
 */
void SplitMergeRenderer::release() {
    glDeletePathsNV(this->fontBase, this->numChars);
}

void SplitMergeRenderer::calcExtents() {
    if (diagram == NULL) {
        return;
    }

    float minX = FLT_MAX, maxX = -FLT_MAX, minY = FLT_MAX, maxY = -FLT_MAX;

    for (int i = 0; i < (int)sortedSeries.Count(); i++) {
        protein_calls::SplitMergeCall::SplitMergeMappable* smm = diagram->GetSeries(sortedSeries[i])->GetMappable();
        vislib::Pair<float, float> p = smm->GetAbscissaRange();
        if (p.First() < minX) {
            minX = p.First();
        }
        if (p.Second() > maxX) {
            maxX = p.Second();
        }

        p = smm->GetOrdinateRange();
        if (p.First() < minY) {
            minY = p.First();
        }
        if (p.Second() > maxY) {
            maxY = p.Second();
        }
    }

    this->bounds.Set(minX, -1.0f - (sortedSeries.Count() - 1) * seriesSpacing, maxX, 1.0f);
}

bool SplitMergeRenderer::GetExtents(mmstd_gl::CallRender2DGL& call) {
    // set the bounding box to 0..1

    if (diagram == NULL) {
        return false;
    }

    call.AccessBoundingBoxes().SetBoundingBox(this->bounds.GetLeft() - noseLength, this->bounds.GetBottom(), 0,
        this->bounds.GetRight() + noseLength, this->bounds.GetTop(), 0);

    return true;
}

/*
 * Callback for mouse events (move, press, and release)
 */
bool SplitMergeRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;

    if (diagram != NULL) {
        if (flags & view::MOUSEFLAG_MODKEY_ALT_DOWN) {
            if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
                if (x > bounds.Left() && x < bounds.Right() && y > bounds.Bottom() && y < bounds.Top()) {
                    float tmp = (y - 1) / -seriesSpacing;
                    int series = static_cast<int>(tmp);
                    if (series >= 0 && series < (int)sortedSeries.Count() && (tmp - static_cast<int>(tmp)) < 0.5f) {
                        // Log::DefaultLog.WriteInfo( "I hit series %s",
                        // diagram->GetSeries(sortedSeries[series])->GetName());
                        consumeEvent = true;
                        selectedSeries = diagram->GetSeries(sortedSeries[series]);
                    }
                }
                if (!consumeEvent) {
                    selectedSeries = NULL;
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
            }
        }
    }

    return consumeEvent;
}


void SplitMergeRenderer::closePath(protein_calls::SplitMergeCall::SplitMergeMappable* smm, int seriesIdx,
    vislib::Array<GLubyte>& cmds, vislib::Array<float>& coords, int idx, int start) {
    float x, y;
    int lastPt = static_cast<int>(coords.Count() - 2);
    cmds.Append(GL_LINE_TO_NV);
    coords.Append(coords[coords.Count() - 2] + noseLength);
    coords.Append(-seriesIdx * seriesSpacing);
    for (int j = lastPt; j >= lastPt - (idx - start - 1) * 2; j -= 2) {
        cmds.Append(GL_LINE_TO_NV);
        x = coords[j];
        y = coords[j + 1];
        y += seriesIdx * seriesSpacing;
        y = -y;
        y -= seriesIdx * seriesSpacing;
        coords.Append(x);
        coords.Append(y);
    }
    cmds.Append(GL_LINE_TO_NV);
    coords.Append(coords[coords.Count() - 2] - noseLength);
    coords.Append(-seriesIdx * seriesSpacing);
    cmds.Append(GL_CLOSE_PATH_NV);
}

/*
 * SplitMergeRenderer::Render
 */
bool SplitMergeRenderer::Render(mmstd_gl::CallRender2DGL& call) {
    // get pointer to Diagram2DCall
    diagram = this->dataCallerSlot.CallAs<protein_calls::SplitMergeCall>();
    if (diagram == NULL)
        return false;
    // execute the call
    if (!(*diagram)(protein_calls::SplitMergeCall::CallForGetData))
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


    glClearStencil(0);
    // glClearColor(0,0,0,0);
    // glStencilMask(~0);
    glClear(GL_STENCIL_BUFFER_BIT);
    glDisable(GL_LINE_WIDTH);

    // array for indirection/sorting/hiding/whatever
    sortedSeries.AssertCapacity(diagram->GetSeriesCount());
    // sortedSeriesInverse.AssertCapacity(diagram->GetSeriesCount());
    // selectionLevel.AssertCapacity(diagram->GetSeriesCount());
    selectionLevel.SetCount(diagram->GetSeriesCount());
    sortedSeries.Clear();
    // sortedSeriesInverse.SetCount(diagram->GetSeriesCount());

    for (int i = 0; i < (int)diagram->GetSeriesCount(); i++) {
        sortedSeries.Add(i);
        // sortedSeriesInverse[i] = i;
        selectionLevel[i] = 0;
    }

#if 0
    vislib::Array<FastMapWrapper> fmps;
    FastMapWrapper::sortedSeries = &sortedSeries;
    FastMapWrapper::diagram = diagram;
    fmps.AssertCapacity(diagram->GetSeriesCount());
    fmps.SetCount(diagram->GetSeriesCount());
    for (int i = 0; i < (int)diagram->GetSeriesCount(); i++) {
        fmps[i].index = i;
    }
    vislib::math::Fastmap<FastMapWrapper, int, 1> fm;
#endif

    if (this->visibilityFromSelection.Param<param::BoolParam>()->Value()) {
        for (int i = 0; i < (int)sortedSeries.Count(); i++) {
            if (diagram->GetSeries(sortedSeries[i]) == selectedSeries) {
                selectionLevel[sortedSeries[i]]++;
            }
        }
        // BUG TODO FIXME WTF
        // WARNING PROPAGATION SLIPS! YOU NEED DOUBLE BUFFERING (selectionLevel) FOR THIS TO WORK!!!
        for (int i = 0; i < this->numVisibilityPropagationRounds.Param<param::IntParam>()->Value(); i++) {
            for (int t = 0; t < (int)diagram->GetTransitionCount(); t++) {
                protein_calls::SplitMergeCall::SplitMergeTransition* smt = diagram->GetTransition(t);
                if (selectionLevel[smt->SourceSeries()] > 0) {
                    selectionLevel[smt->DestinationSeries()]++;
                }
                if (selectionLevel[smt->DestinationSeries()] > 0) {
                    selectionLevel[smt->SourceSeries()]++;
                }
            }
        }
        for (int i = (int)sortedSeries.Count() - 1; i >= 0; i--) {
            if (selectionLevel[sortedSeries[i]] == 0) {
                seriesVisible[sortedSeries[i]] = false;
            } else {
                seriesVisible[sortedSeries[i]] = true;
            }
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
    }
    for (int i = (int)sortedSeries.Count() - 1; i >= 0; i--) {
        if (!seriesVisible[sortedSeries[i]]) {
            // sortedSeriesInverse[sortedSeries[i]] = -1;
            sortedSeries.RemoveAt(i);
        }
    }
    // TERROR-CODE
    // sortedSeries.Insert(sortedSeries.Count() / 2, sortedSeries.Count() - 1);

    this->calcExtents();

    maxY = -FLT_MAX;
    for (int i = 0; i < (int)sortedSeries.Count(); i++) {
        float tmpY = diagram->GetSeries(sortedSeries[i])->GetMappable()->GetOrdinateRange().GetSecond();
        if (tmpY > maxY) {
            maxY = tmpY;
        }
    }

    GLuint pathBase = glGenPathsNV((GLsizei)sortedSeries.Count());
    for (int i = 0; i < (int)sortedSeries.Count(); i++) {
        protein_calls::SplitMergeCall::SplitMergeSeries* sms = diagram->GetSeries(sortedSeries[i]);
        protein_calls::SplitMergeCall::SplitMergeMappable* smm = sms->GetMappable();
        int increment = 50;
        vislib::Array<GLubyte> cmds;
        cmds.SetCapacityIncrement(increment);
        vislib::Array<float> coords;
        coords.SetCapacityIncrement(increment);

        int start = 0;
        int end = start;
        int count = smm->GetDataCount();
        int idx = 0;
        float x, y;
        bool ret;
        bool open = false;

        while (idx < count) {
            ret = smm->GetAbscissaValue(idx, &x);
            if (ret) {
                open = true;
                y = smm->GetOrdinateValue(idx) / maxY - i * seriesSpacing;
                if (idx == start) {
                    cmds.Append(GL_MOVE_TO_NV);
                } else {
                    cmds.Append(GL_LINE_TO_NV);
                }
                coords.Append(x);
                coords.Append(y);
            } else {
                if (idx > start) {
                    // we have something to close
                    closePath(smm, i, cmds, coords, idx, start);
                    open = false;
                }
                start = idx + 1;
            }
            idx++;
        }

        if (open) {
            closePath(smm, i, cmds, coords, idx, start);
        }

        glPathCommandsNV(pathBase + i, (GLsizei)cmds.Count(), cmds.PeekElements(), (GLsizei)coords.Count(), GL_FLOAT,
            coords.PeekElements());

        glStencilFillPathNV(pathBase + i, GL_COUNT_UP_NV, 0x1F);
    }

    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

    for (int i = 0; i < (int)sortedSeries.Count(); i++) {
        if (selectedSeries == NULL || *selectedSeries == *diagram->GetSeries(sortedSeries[i])) {
            ::glColor3fv(diagram->GetSeries(sortedSeries[i])->GetColorRGB().PeekComponents());
        } else {
            ::glColor4fv(unselectedColor.PeekComponents());
        }

        // glColor3fv(diagram->GetSeries(sortedSeries[i])->GetColorRGB().PeekComponents());
        glCoverFillPathNV(pathBase + i, GL_BOUNDING_BOX_NV);
    }
    glDeletePathsNV(pathBase, (GLsizei)sortedSeries.Count());

    // glDisable(GL_STENCIL_TEST);

    // TODO: draw splines

    // TODO: draw time ruler

    // TODO: series indirection for sorting/hiding unselected stuff

    // TODO: selection with transitive affection (selected, repeat {affect connected})

    pathBase = glGenPathsNV((GLsizei)diagram->GetTransitionCount());
    vislib::Array<GLubyte> cmds;
    cmds.AssertCapacity(5);
    cmds.Append(GL_MOVE_TO_NV);
    cmds.Append(GL_CUBIC_CURVE_TO_NV);
    cmds.Append(GL_LINE_TO_NV);
    cmds.Append(GL_CUBIC_CURVE_TO_NV);
    cmds.Append(GL_CLOSE_PATH_NV);
    vislib::Array<float> coords;
    coords.AssertCapacity(2 + 6 + 2 + 6);
    coords.SetCount(2 + 6 + 2 + 6);
    float srcX, srcYTop, srcYBottom, dstX, dstYTop, dstYBottom;
    unsigned int counter = 0;
    unsigned int visibleSeriesIdx;

    // count how many series are invisible ahead of the current one
    unsigned int* invisibleCounter = new unsigned int[diagram->GetSeriesCount()];
    if (diagram->GetSeriesCount() > 0) {
        invisibleCounter[0] = 0;
        for (unsigned int i = 1; i < diagram->GetSeriesCount(); i++) {
            invisibleCounter[i] = invisibleCounter[i - 1] + (seriesVisible[i - 1] ? 0 : 1);
        }
    }

    GLfloat gradient[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};

    for (int i = 0; i < (int)diagram->GetTransitionCount(); i++) {
        // coords.Clear();
        protein_calls::SplitMergeCall::SplitMergeTransition* smt = diagram->GetTransition(i);
        if (!seriesVisible[smt->SourceSeries()] || !seriesVisible[smt->DestinationSeries()]) {
            continue;
        }

        // COORDS

        unsigned int srcSeriesIdx = smt->SourceSeries();
        protein_calls::SplitMergeCall::SplitMergeSeries* srcSeries = this->diagram->GetSeries(srcSeriesIdx);
        protein_calls::SplitMergeCall::SplitMergeMappable* srcSMM = srcSeries->GetMappable();
        unsigned int srcDataIdx = smt->SourceSeriesDataIndex();
        srcSMM->GetAbscissaValue(srcDataIdx, &srcX);
        visibleSeriesIdx = srcSeriesIdx - invisibleCounter[srcSeriesIdx];
        srcYTop = srcSMM->GetOrdinateValue(srcDataIdx) / maxY - visibleSeriesIdx * seriesSpacing;
        srcYBottom = (-srcSMM->GetOrdinateValue(srcDataIdx) / maxY) - visibleSeriesIdx * seriesSpacing;

        unsigned int dstSeriesIdx = smt->DestinationSeries();
        protein_calls::SplitMergeCall::SplitMergeSeries* dstSeries = this->diagram->GetSeries(dstSeriesIdx);
        protein_calls::SplitMergeCall::SplitMergeMappable* dstSMM = dstSeries->GetMappable();
        unsigned int dstDataIdx = smt->DestinationSeriesDataIndex();
        dstSMM->GetAbscissaValue(dstDataIdx, &dstX);
        visibleSeriesIdx = dstSeriesIdx - invisibleCounter[dstSeriesIdx];
        dstYTop = dstSMM->GetOrdinateValue(dstDataIdx) / maxY - visibleSeriesIdx * seriesSpacing;
        dstYBottom = (-dstSMM->GetOrdinateValue(dstDataIdx) / maxY) - visibleSeriesIdx * seriesSpacing;

        // GL_MOVE_TO_NV
        coords[0] = srcX;
        coords[1] = srcYTop;
        // GL_CUBIC_CURVE_TO_NV
        // coords.Append(dstX);
        coords[2] = srcX + (dstX - srcX) * 0.75f;
        coords[3] = srcYTop;
        coords[4] = dstX - (dstX - srcX) * 0.25f;
        coords[5] = dstYTop;
        coords[6] = dstX;
        coords[7] = dstYTop;
        // GL_LINE_TO_NV
        coords[8] = dstX;
        coords[9] = dstYBottom;
        // GL_CUBIC_CURVE_TO_NV
        // coords.Append(srcX);
        coords[10] = dstX - (dstX - srcX) * 0.75f;
        coords[11] = dstYBottom;
        coords[12] = srcX + (dstX - srcX) * 0.25f;
        coords[13] = srcYBottom;
        coords[14] = srcX;
        coords[15] = srcYBottom;

        glPathCommandsNV(pathBase + counter, (GLsizei)cmds.Count(), cmds.PeekElements(), (GLsizei)coords.Count(),
            GL_FLOAT, coords.PeekElements());
        glStencilFillPathNV(pathBase + counter, GL_COUNT_UP_NV, 0x1F);

        const float* topCol;
        const float* bottomCol;
        // TODO selection level

        if (selectedSeries == NULL || *selectedSeries == *srcSeries) {
            topCol = srcSeries->GetColorRGB().PeekComponents();
        } else {
            topCol = unselectedColor.PeekComponents();
        }
        if (selectedSeries == NULL || *selectedSeries == *dstSeries) {
            bottomCol = dstSeries->GetColorRGB().PeekComponents();
        } else {
            bottomCol = unselectedColor.PeekComponents();
        }

        if (srcSeriesIdx > dstSeriesIdx) {
            const float* tmp = topCol;
            topCol = bottomCol;
            bottomCol = tmp;
        }

        // if (srcSeriesIdx < dstSeriesIdx) {
        //    topCol = srcSeries->GetColorRGB().PeekComponents();
        //    bottomCol = dstSeries->GetColorRGB().PeekComponents();
        //} else {
        //    bottomCol = srcSeries->GetColorRGB().PeekComponents();
        //    topCol = dstSeries->GetColorRGB().PeekComponents();
        //}
        gradient[0][2] = bottomCol[0];
        gradient[1][2] = bottomCol[1];
        gradient[2][2] = bottomCol[2];
        gradient[0][1] = topCol[0] - bottomCol[0];
        gradient[1][1] = topCol[1] - bottomCol[1];
        gradient[2][1] = topCol[2] - bottomCol[2];

        // if (selectedSeries == NULL || *selectedSeries == *diagram->GetSeries(sortedSeries[i])) {
        //    ::glColor3fv(diagram->GetSeries(sortedSeries[i])->GetColorRGB().PeekComponents());
        //} else {
        //    ::glColor4fv(unselectedColor.PeekComponents());
        //}

        // glColor3fv(diagram->GetSeries(sortedSeries[i])->GetColorRGB().PeekComponents());
        //::glColor4fv(unselectedColor.PeekComponents());
        glPathColorGenNV(GL_PRIMARY_COLOR, GL_PATH_OBJECT_BOUNDING_BOX_NV, GL_RGB, &gradient[0][0]);
        glCoverFillPathNV(pathBase + counter, GL_BOUNDING_BOX_NV);

        counter++;
    }
    delete[] invisibleCounter;

    // counter = 0;
    // for (int i = 0; i < diagram->GetTransitionCount(); i++) {
    //    SplitMergeCall::SplitMergeTransition *smt = diagram->GetTransition(i);
    //    if (!seriesVisible[smt->SourceSeries()] || !seriesVisible[smt->DestinationSeries()]) {
    //        continue;
    //    }
    //    const float *topCol;
    //    const float *bottomCol;
    //    // TODO selection level
    //    if (smt->SourceSeries() < smt->DestinationSeries()) {
    //        topCol = diagram->GetSeries(smt->SourceSeries())->GetColorRGB().PeekComponents();
    //        bottomCol = diagram->GetSeries(smt->DestinationSeries())->GetColorRGB().PeekComponents();
    //    } else {
    //        bottomCol = diagram->GetSeries(smt->SourceSeries())->GetColorRGB().PeekComponents();
    //        topCol = diagram->GetSeries(smt->DestinationSeries())->GetColorRGB().PeekComponents();
    //    }
    //    gradient[0][2] = bottomCol[0];
    //    gradient[1][2] = bottomCol[1];
    //    gradient[2][2] = bottomCol[2];
    //    gradient[0][1] = topCol[0] - bottomCol[0];
    //    gradient[1][1] = topCol[1] - bottomCol[1];
    //    gradient[2][1] = topCol[2] - bottomCol[2];

    //    //if (selectedSeries == NULL || *selectedSeries == *diagram->GetSeries(sortedSeries[i])) {
    //    //    ::glColor3fv(diagram->GetSeries(sortedSeries[i])->GetColorRGB().PeekComponents());
    //    //} else {
    //    //    ::glColor4fv(unselectedColor.PeekComponents());
    //    //}

    //    //glColor3fv(diagram->GetSeries(sortedSeries[i])->GetColorRGB().PeekComponents());
    //    //::glColor4fv(unselectedColor.PeekComponents());
    //    glPathColorGenNV(GL_PRIMARY_COLOR, GL_PATH_OBJECT_BOUNDING_BOX_NV, GL_RGB, &gradient[0][0]);
    //    glCoverFillPathNV(pathBase + counter, GL_BOUNDING_BOX_NV);
    //    counter++;
    //}
    glDeletePathsNV(pathBase, (GLsizei)diagram->GetTransitionCount());

    glPathColorGenNV(GL_PRIMARY_COLOR, GL_NONE, GL_NONE, NULL);

    glStencilFunc(GL_NOTEQUAL, 0, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
    glColor3fv(this->fgColor.PeekComponents());

    for (int i = 0; i < (int)sortedSeries.Count(); i++) {
        vislib::StringA theName = diagram->GetSeries(sortedSeries[i])->GetName();
        theName.Append("x");
        GLfloat* kerning = new GLfloat[theName.Length()];
        kerning[0] = 0.0f; // Initial glyph offset is zero

        glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV, theName.Length(), GL_UNSIGNED_BYTE, theName.PeekBuffer(),
            this->fontBase, 1.0f, 1.0f, GL_TRANSLATE_X_NV, kerning + 1);

        glPushMatrix();
        glTranslatef(
            -kerning[theName.Length() - 1] - this->fontSize * 0.5f, -i * seriesSpacing - this->fontSize * 0.25f, 0.0f);

        glStencilFillPathInstancedNV(theName.Length() - 1, GL_UNSIGNED_BYTE, theName.PeekBuffer(), fontBase,
            GL_PATH_FILL_MODE_NV, 0xFF, GL_TRANSLATE_X_NV, kerning);

        glCoverFillPathInstancedNV(theName.Length() - 1, GL_UNSIGNED_BYTE, theName.PeekBuffer(), fontBase,
            GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV, GL_TRANSLATE_X_NV, kerning);

        glPopMatrix();

        delete[] kerning;
    }

    glDisable(GL_STENCIL_TEST);

    float decorationDepth = 0.0f;
    if (this->showGuidesParam.Param<param::BoolParam>()->Value()) {
        for (int i = 0; i < (int)diagram->GetGuideCount(); i++) {
            protein_calls::SplitMergeCall::SplitMergeGuide* g = diagram->GetGuide(i);
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
            case protein_calls::SplitMergeCall::SPLITMERGE_GUIDE_HORIZONTAL:
                /*pos = g->GetPosition() - yRange.First();
                pos /= yRange.GetSecond() - yRange.GetFirst();*/
                pos = g->GetPosition();
                ::glVertex3f(0.0f, pos, decorationDepth);
                ::glVertex3f(this->bounds.GetRight(), pos, decorationDepth);
                ::glEnd();
                ::glDisable(GL_LINE_STIPPLE);
                // theFont.DrawString(aspect, pos, fontSize * 0.5f, true,
                //    tmpString.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
                break;
            case protein_calls::SplitMergeCall::SPLITMERGE_GUIDE_VERTICAL:
                // pos = g->GetPosition() - xRange.First();
                // pos /= xRange.GetSecond() - xRange.GetFirst();
                // pos *= aspect;
                pos = g->GetPosition();
                ::glVertex3f(pos, this->bounds.GetBottom(), decorationDepth);
                ::glVertex3f(pos, this->bounds.GetTop(), decorationDepth);
                ::glEnd();
                ::glDisable(GL_LINE_STIPPLE);
                // theFont.DrawString(pos, 1.0f, fontSize * 0.5f, true,
                //    tmpString.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);
                break;
            }
        }
    }

    return true;
}
