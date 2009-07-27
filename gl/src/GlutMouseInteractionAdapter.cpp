/*
 * GlutMouseInteractionAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/GlutMouseInteractionAdapter.h"

#include "vislibGlutInclude.h"


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::GlutMouseInteractionAdapter
 */
vislib::graphics::gl::GlutMouseInteractionAdapter::GlutMouseInteractionAdapter(
        const SmartPtr<CameraParameters>& params, const unsigned int cntButtons) 
        : mia(params, cntButtons) {
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::~GlutMouseInteractionAdapter
 */
vislib::graphics::gl::GlutMouseInteractionAdapter::~GlutMouseInteractionAdapter(
        void) {
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::OnKeyDown
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::OnKeyDown(
        const unsigned char key, const int x, const int y) {
    this->mia.SetMousePosition(x, y, true);
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::OnMouseButton
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::OnMouseButton(
        const int button, const int state, const int x, const int y) {
    MouseInteractionAdapter::Button btn = MouseInteractionAdapter::BUTTON_LEFT;

    switch (button) {
        case GLUT_LEFT_BUTTON: 
            btn = MouseInteractionAdapter::BUTTON_LEFT; 
            break;

        case GLUT_RIGHT_BUTTON: 
            btn = MouseInteractionAdapter::BUTTON_RIGHT; 
            break;

        case GLUT_MIDDLE_BUTTON: 
            btn = MouseInteractionAdapter::BUTTON_MIDDLE;
            break;
    }

    this->mia.SetMousePosition(x, y, true);
    this->mia.SetMouseButtonState(btn, (state == GLUT_DOWN));
    this->setModifierState(::glutGetModifiers());
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::OnMouseMove
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::OnMouseMove(
        const int x, const int y) {
    this->mia.SetMousePosition(x, y, true);
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::OnMousePassiveMove
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::OnMousePassiveMove(
        const int x, const int y) {
    this->mia.SetMousePosition(x, y, true);
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::OnResize
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::OnResize(
        const int width, const int height) {
    SmartPtr<CameraParameters> params = this->mia.GetCamera();

    if (!params.IsNull()) {
        params->SetVirtualViewSize(static_cast<ImageSpaceType>(width),
            static_cast<ImageSpaceType>(height));
    }
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::OnSpecialKeyDown
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::OnSpecialKeyDown(
        const int key, const int x, const int y) {
    this->mia.SetMousePosition(x, y, true);
    this->setModifierState(::glutGetModifiers());
}


/*
 * vislib::graphics::gl::GlutMouseInteractionAdapter::setModifierState
 */
void vislib::graphics::gl::GlutMouseInteractionAdapter::setModifierState(
        const int glutModifiers) {
    this->mia.SetModifierState(InputModifiers::MODIFIER_SHIFT, 
        (glutModifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->mia.SetModifierState(InputModifiers::MODIFIER_CTRL, 
        (glutModifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->mia.SetModifierState(InputModifiers::MODIFIER_ALT, 
        (glutModifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
}
