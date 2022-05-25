/*
 * AbstractOverrideView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore_gl/view/AbstractOverrideViewGL.h"

using namespace megamol::core_gl;
using namespace megamol::frontend_resources;


/*
 * view::AbstractOverrideView::AbstractOverrideView
 */
view::AbstractOverrideViewGL::AbstractOverrideViewGL(void)
        : renderViewSlot("renderView", "Slot for outgoing rendering requests to other views")
        , viewportWidth(1)
        , viewportHeight(1) {

    this->renderViewSlot.SetCompatibleCall<view::CallRenderViewGLDescription>();
    this->MakeSlotAvailable(&this->renderViewSlot);
}


/*
 * view::AbstractOverrideView::~AbstractOverrideView
 */
view::AbstractOverrideViewGL::~AbstractOverrideViewGL(void) {

    // TODO: Implement
}


/*
 * view::AbstractOverrideView::DefaultTime
 */
float view::AbstractOverrideViewGL::DefaultTime(double instTime) const {
    view::CallRenderViewGL* call =
        const_cast<view::AbstractOverrideViewGL*>(this)->renderViewSlot.CallAs<view::CallRenderViewGL>();
    if (call == NULL)
        return 0.0f;
    const core::CalleeSlot* s = call->PeekCalleeSlot();
    if (s == NULL)
        return 0.0f;
    const Module* m = static_cast<const Module*>(s->Owner());
    if (m == NULL)
        return 0.0f;
    const AbstractView* v = dynamic_cast<const AbstractView*>(m);
    if (v == NULL)
        return 0.0f;
    return v->DefaultTime(instTime);
}


/*
 * view::AbstractOverrideView::SerialiseCamera
 */
void view::AbstractOverrideViewGL::SerialiseCamera(vislib::Serialiser& serialiser) const {
    // TODO: Implement
}


/*
 * view::AbstractOverrideView::DeserialiseCamera
 */
void view::AbstractOverrideViewGL::DeserialiseCamera(vislib::Serialiser& serialiser) {
    // TODO: Implement
}


/*
 * view::AbstractOverrideView::ResetView
 */
void view::AbstractOverrideViewGL::ResetView(void) {
    // resets camera, not override values
    view::CallRenderViewGL* crv = this->getCallRenderView();
    if (crv != NULL) {
        (*crv)(view::CallRenderViewGL::CALL_RESETVIEW);
    }
}


/*
 * view::AbstractOverrideView::Resize
 */
void view::AbstractOverrideViewGL::Resize(unsigned int width, unsigned int height) {
    this->viewportWidth = width;
    if (this->viewportWidth < 1)
        this->viewportWidth = 1;
    this->viewportHeight = height;
    if (this->viewportHeight < 1)
        this->viewportHeight = 1;
}


///*
// * view::AbstractOverrideView::SetCursor2DButtonState
// */
//void view::AbstractOverrideView::SetCursor2DButtonState(unsigned int btn, bool down) {
//    view::CallRenderViewGL *crv = this->getCallRenderView();
//    if (crv != NULL) {
//        crv->SetMouseButton(btn, down);
//        (*crv)(view::CallRenderViewGL::CALL_SETCURSOR2DBUTTONSTATE);
//    }
//}
//
//
///*
// * view::AbstractOverrideView::SetCursor2DPosition
// */
//void view::AbstractOverrideView::SetCursor2DPosition(float x, float y) {
//    view::CallRenderViewGL *crv = this->getCallRenderView();
//    if (crv != NULL) {
//        this->packMouseCoordinates(x, y);
//        crv->SetMousePosition(x, y);
//        (*crv)(view::CallRenderViewGL::CALL_SETCURSOR2DPOSITION);
//    }
//}
//
//
///*
// * view::AbstractOverrideView::SetInputModifier
// */
//void view::AbstractOverrideView::SetInputModifier(view::Modifier mod, bool down) {
//    view::CallRenderViewGL *crv = this->getCallRenderView();
//    if (crv != NULL) {
//        crv->SetInputModifier(mod, down);
//        (*crv)(view::CallRenderViewGL::CALL_SETINPUTMODIFIER);
//    }
//}


bool view::AbstractOverrideViewGL::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderViewGL::FnOnKey))
        return false;

    return true;
}


bool view::AbstractOverrideViewGL::OnChar(unsigned int codePoint) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderViewGL::FnOnChar))
        return false;

    return true;
}


bool view::AbstractOverrideViewGL::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::MouseButton;
    evt.mouseButtonData.button = button;
    evt.mouseButtonData.action = action;
    evt.mouseButtonData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderViewGL::FnOnMouseButton))
        return false;

    return true;
}


bool view::AbstractOverrideViewGL::OnMouseMove(double x, double y) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::MouseMove;
    evt.mouseMoveData.x = x;
    evt.mouseMoveData.y = y;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderViewGL::FnOnMouseMove))
        return false;

    return true;
}


bool view::AbstractOverrideViewGL::OnMouseScroll(double dx, double dy) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderViewGL::FnOnMouseScroll))
        return false;

    return true;
}


/*
 * view::AbstractOverrideView::disconnectOutgoingRenderCall
 */
void view::AbstractOverrideViewGL::disconnectOutgoingRenderCall(void) {
    this->renderViewSlot.ConnectCall(NULL);
}


/*
 * view::AbstractOverrideView::getConnectedView
 */
megamol::core::view::AbstractView* view::AbstractOverrideViewGL::getConnectedView(void) const {
    core::Call* c = const_cast<core::CallerSlot*>(&this->renderViewSlot)->CallAs<core::Call>();
    if ((c == NULL) || (c->PeekCalleeSlot() == NULL))
        return NULL;
    return const_cast<core::view::AbstractView*>(
        dynamic_cast<const core::view::AbstractView*>(c->PeekCalleeSlot()->Parent().get()));
}


/*
 * view::AbstractOverrideView::packMouseCoordinates
 */
void view::AbstractOverrideViewGL::packMouseCoordinates(float& x, float& y) {
    // intentionally empty
    // do something smart in the derived classes
}
