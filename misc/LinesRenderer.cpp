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

        if (l.ColourArray() == NULL) {
            if (useColourArray) {
                useColourArray = false;
                ::glDisableClientState(GL_COLOR_ARRAY);
            }
            ::glColor3ubv(l.GlobalColour().PeekComponentes());
        } else {
            if (!useColourArray) {
                useColourArray = true;
                ::glEnableClientState(GL_COLOR_ARRAY);
            }
            ::glColorPointer(l.HasColourAlpha() ? 4 : 3,
                l.IsFloatColour() ? GL_FLOAT : GL_UNSIGNED_BYTE, 0, 
                l.ColourArray());
        }

        ::glVertexPointer(3, GL_FLOAT, 0, l.VertexArray());

        if (l.IndexArray() == NULL) {
            ::glDrawArrays(GL_LINES, 0, l.Count());
        } else {
            ::glDrawElements(GL_LINES, l.Count(),
                GL_UNSIGNED_INT, l.IndexArray());
        }
    }

    if (useColourArray) ::glDisableClientState(GL_COLOR_ARRAY);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return true;
}
