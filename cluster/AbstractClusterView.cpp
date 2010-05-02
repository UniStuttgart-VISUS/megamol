/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/AbstractClusterView.h"
#include "cluster/InfoIconRenderer.h"
#include <GL/gl.h>
#include "vislib/sysfunctions.h"


using namespace megamol::core;


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void)
        : view::AbstractTileView(), ClusterControllerClient() {

    // slot initialized in 'ClusterControllerClient::ctor'
    this->MakeSlotAvailable(&this->registerSlot);

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::~AbstractClusterView
 */
cluster::AbstractClusterView::~AbstractClusterView(void) {

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::ResetView
 */
void cluster::AbstractClusterView::ResetView(void) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DButtonState
 */
void cluster::AbstractClusterView::SetCursor2DButtonState(unsigned int btn, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DPosition
 */
void cluster::AbstractClusterView::SetCursor2DPosition(float x, float y) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetInputModifier
 */
void cluster::AbstractClusterView::SetInputModifier(mmcInputModifier mod, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::renderFallbackView
 */
void cluster::AbstractClusterView::renderFallbackView(void) {

    ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());
    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    if ((this->getViewportHeight() <= 1) || (this->getViewportWidth() <= 1)) return;

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    float aspect = static_cast<float>(this->getViewportWidth())
        / static_cast<float>(this->getViewportHeight());
    if ((this->getProjType() == vislib::graphics::CameraParameters::MONO_PERSPECTIVE)
            || (this->getProjType() == vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC)) {
        if (aspect > 1.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(2.0f, -2.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    } else {
        if (this->getEye() == vislib::graphics::CameraParameters::RIGHT_EYE) {
            ::glTranslatef(0.5f, 0.0f, 0.0f);
        } else {
            ::glTranslatef(-0.5f, 0.0f, 0.0f);
        }
        if (aspect > 2.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(1.0f, -1.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    }
    const float border = 0.05f;
    ::glTranslatef(border, border, 0.0f);
    ::glScalef(1.0f - 2.0f * border, 1.0f - 2.0f * border, 0.0f);

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    InfoIconRenderer::IconState icon = InfoIconRenderer::ICONSTATE_UNKNOWN;
    vislib::TString msg;
    this->getFallbackMessageInfo(msg, icon);
    InfoIconRenderer::RenderInfoIcon(icon, msg);

}


/*
 * cluster::AbstractClusterView::getFallbackMessageInfo
 */
void cluster::AbstractClusterView::getFallbackMessageInfo(vislib::TString& outMsg,
        InfoIconRenderer::IconState& outState) {
    outState = InfoIconRenderer::ICONSTATE_UNKNOWN;
    outMsg = _T("State unknown");

    // TODO: Build real message and icon state
    int t = vislib::sys::GetTicksOfDay() % 50000 / 10000;
    t = 0;
    outState = static_cast<InfoIconRenderer::IconState>(t);
    switch (t) {
        case 0:
            outMsg = _T("DEBUG: I don't know what I am doing");
            break;
        case 1:
            outMsg = _T("DEBUG: I don't feel so good");
            break;
        case 2:
            outMsg = _T("DEBUG: Everything looks fine from here");
            break;
        case 3:
            outMsg = _T("DEBUG: Give me some time");
            break;
        case 4:
            outMsg = _T("DEBUG: Working for you ...");
            break;
    }
}
