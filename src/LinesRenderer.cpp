/*
 * LinesRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
//#define _USE_MATH_DEFINES
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "LinesRenderer.h"
#include "TriSoup/LinesDataCall.h"
#include "mmcore/view/CallRender3D.h"
//#include <cmath>

using namespace megamol;


/*
 * trisoup::LinesRenderer::LinesRenderer
 */
trisoup::LinesRenderer::LinesRenderer(void) : core::view::Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source") {

    this->getDataSlot.SetCompatibleCall<LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * trisoup::LinesRenderer::~LinesRenderer
 */
trisoup::LinesRenderer::~LinesRenderer(void) {
    this->Release();
}


/*
 * trisoup::LinesRenderer::create
 */
bool trisoup::LinesRenderer::create(void) {
    // intentionally empty
    return true;
}


/*
 * trisoup::LinesRenderer::GetCapabilities
 */
bool trisoup::LinesRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        core::view::CallRender3D::CAP_RENDER
        );

    return true;
}


/*
 * trisoup::LinesRenderer::GetExtents
 */
bool trisoup::LinesRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    LinesDataCall *ldc = this->getDataSlot.CallAs<LinesDataCall>();
    if (ldc != nullptr) ldc->SetFrameID(static_cast<int>(cr->Time()), true);
    if ((ldc != NULL) && ((*ldc)(1))) {
        cr->SetTimeFramesCount(ldc->FrameCount());
        cr->AccessBoundingBoxes() = ldc->AccessBoundingBoxes();
        cr->AccessBoundingBoxes().MakeScaledWorld(1.0f); // lol-o-mat

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();

    }

    return true;
}


/*
 * trisoup::LinesRenderer::release
 */
void trisoup::LinesRenderer::release(void) {
    // intentionally empty
}


/*
 * trisoup::LinesRenderer::Render
 */
bool trisoup::LinesRenderer::Render(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    ::glDisable(GL_TEXTURE);
    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    ::glLineWidth(1.0f);

    LinesDataCall *ldc = this->getDataSlot.CallAs<LinesDataCall>();
    if (ldc != nullptr) ldc->SetFrameID(static_cast<int>(cr->Time()), true);
    if ((ldc == NULL) || (!(*ldc)(0))) return false;

    bool useColourArray = false;
    ::glEnableClientState(GL_VERTEX_ARRAY);

    for (unsigned int i = 0; i < ldc->Count(); i++) {
        const LinesDataCall::Lines& l = ldc->GetLines()[i];

        if (l.ColourArrayType() == LinesDataCall::Lines::CDT_NONE) {
            if (useColourArray) {
                useColourArray = false;
                ::glDisableClientState(GL_COLOR_ARRAY);
            }
            ::glColor3ubv(l.GlobalColour().PeekComponents());
        } else {
            if (!useColourArray) {
                useColourArray = true;
                ::glEnableClientState(GL_COLOR_ARRAY);
            }
            switch (l.ColourArrayType()) {
                case LinesDataCall::Lines::CDT_BYTE_RGB:
                    ::glColorPointer(3, GL_UNSIGNED_BYTE, 0, l.ColourArrayByte());
                    break;
                case LinesDataCall::Lines::CDT_BYTE_RGBA:
                    ::glColorPointer(4, GL_UNSIGNED_BYTE, 0, l.ColourArrayByte());
                    break;
                case LinesDataCall::Lines::CDT_FLOAT_RGB:
                    ::glColorPointer(3, GL_FLOAT, 0, l.ColourArrayFloat());
                    break;
                case LinesDataCall::Lines::CDT_FLOAT_RGBA:
                    ::glColorPointer(4, GL_FLOAT, 0, l.ColourArrayFloat());
                    break;
                case LinesDataCall::Lines::CDT_DOUBLE_RGB:
                    ::glColorPointer(3, GL_DOUBLE, 0, l.ColourArrayDouble());
                    break;
                case LinesDataCall::Lines::CDT_DOUBLE_RGBA:
                    ::glColorPointer(4, GL_DOUBLE, 0, l.ColourArrayDouble());
                    break;
                default: continue;
            }
        }

        switch (l.VertexArrayDataType()) {
            case LinesDataCall::Lines::DT_FLOAT:
                ::glVertexPointer(3, GL_FLOAT, 0, l.VertexArrayFloat());
                break;
            case LinesDataCall::Lines::DT_DOUBLE:
                ::glVertexPointer(3, GL_DOUBLE, 0, l.VertexArrayDouble());
                break;
            default: continue;
        }

        if (l.IndexArrayDataType() == LinesDataCall::Lines::DT_NONE) {
            ::glDrawArrays(GL_LINES, 0, l.Count());
        } else {
            switch (l.IndexArrayDataType()) {
                case LinesDataCall::Lines::DT_BYTE:
                    ::glDrawElements(GL_LINES, l.Count(), GL_UNSIGNED_BYTE, l.IndexArrayByte());
                    break;
                case LinesDataCall::Lines::DT_UINT16:
                    ::glDrawElements(GL_LINES, l.Count(), GL_UNSIGNED_SHORT, l.IndexArrayUInt16());
                    break;
                case LinesDataCall::Lines::DT_UINT32:
                    ::glDrawElements(GL_LINES, l.Count(), GL_UNSIGNED_INT, l.IndexArrayUInt32());
                    break;
                default: continue;
            }
        }
    }

    if (useColourArray) ::glDisableClientState(GL_COLOR_ARRAY);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return true;
}
