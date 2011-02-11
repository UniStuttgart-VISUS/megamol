/*
 * LinesRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
//#define _USE_MATH_DEFINES
#include "LinesRenderer.h"
#include "LinesDataCall.h"
#include "view/CallRender3D.h"
#include <GL/gl.h>
//#include <cmath>

using namespace megamol::core;


/*
 * misc::LinesRenderer::LinesRenderer
 */
misc::LinesRenderer::LinesRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source") {

    this->getDataSlot.SetCompatibleCall<misc::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * misc::LinesRenderer::~LinesRenderer
 */
misc::LinesRenderer::~LinesRenderer(void) {
    this->Release();
}


/*
 * misc::LinesRenderer::create
 */
bool misc::LinesRenderer::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::LinesRenderer::GetCapabilities
 */
bool misc::LinesRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        );

    return true;
}


/*
 * misc::LinesRenderer::GetExtents
 */
bool misc::LinesRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    LinesDataCall *ldc = this->getDataSlot.CallAs<misc::LinesDataCall>();
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
 * misc::LinesRenderer::release
 */
void misc::LinesRenderer::release(void) {
    // intentionally empty
}


/*
 * misc::LinesRenderer::Render
 */
bool misc::LinesRenderer::Render(Call& call) {

    ::glDisable(GL_TEXTURE);
    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    ::glLineWidth(1.0f);

    LinesDataCall *ldc = this->getDataSlot.CallAs<misc::LinesDataCall>();
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
