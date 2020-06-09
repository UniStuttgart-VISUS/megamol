/*
 * AbstractOverrideView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractOverrideView.h"

using namespace megamol::core;
using namespace megamol::input_events;


/*
 * view::AbstractOverrideView::AbstractOverrideView
 */
view::AbstractOverrideView::AbstractOverrideView(void) : AbstractView(),
        renderViewSlot("renderView", "Slot for outgoing rendering requests to other views"),
        viewportWidth(1), viewportHeight(1) {

    this->renderViewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->renderViewSlot);

}


/*
 * view::AbstractOverrideView::~AbstractOverrideView
 */
view::AbstractOverrideView::~AbstractOverrideView(void) {

    // TODO: Implement

}


/*
 * view::AbstractOverrideView::DefaultTime
 */
float view::AbstractOverrideView::DefaultTime(double instTime) const {
    view::CallRenderView *call = const_cast<view::AbstractOverrideView*>(this)->renderViewSlot.CallAs<view::CallRenderView>();
    if (call == NULL) return 0.0f;
    const CalleeSlot *s = call->PeekCalleeSlot();
    if (s == NULL) return 0.0f;
    const Module *m = static_cast<const Module*>(s->Owner());
    if (m == NULL) return 0.0f;
    const AbstractView *v = dynamic_cast<const AbstractView *>(m);
    if (v == NULL) return 0.0f;
    return v->DefaultTime(instTime);
}


/*
 * view::AbstractOverrideView::GetCameraSyncNumber
 */
unsigned int view::AbstractOverrideView::GetCameraSyncNumber(void) const {
    return 0; // TODO: Implement
}


/*
 * view::AbstractOverrideView::SerialiseCamera
 */
void view::AbstractOverrideView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    // TODO: Implement
}


/*
 * view::AbstractOverrideView::DeserialiseCamera
 */
void view::AbstractOverrideView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    // TODO: Implement
}


/*
 * view::AbstractOverrideView::ResetView
 */
void view::AbstractOverrideView::ResetView(void) {
    // resets camera, not override values
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        (*crv)(view::CallRenderView::CALL_RESETVIEW);
    }
}


/*
 * view::AbstractOverrideView::Resize
 */
void view::AbstractOverrideView::Resize(unsigned int width, unsigned int height) {
    this->viewportWidth = width;
    if (this->viewportWidth < 1) this->viewportWidth = 1;
    this->viewportHeight = height;
    if (this->viewportHeight < 1) this->viewportHeight = 1;
}


///*
// * view::AbstractOverrideView::SetCursor2DButtonState
// */
//void view::AbstractOverrideView::SetCursor2DButtonState(unsigned int btn, bool down) {
//    view::CallRenderView *crv = this->getCallRenderView();
//    if (crv != NULL) {
//        crv->SetMouseButton(btn, down);
//        (*crv)(view::CallRenderView::CALL_SETCURSOR2DBUTTONSTATE);
//    }
//}
//
//
///*
// * view::AbstractOverrideView::SetCursor2DPosition
// */
//void view::AbstractOverrideView::SetCursor2DPosition(float x, float y) {
//    view::CallRenderView *crv = this->getCallRenderView();
//    if (crv != NULL) {
//        this->packMouseCoordinates(x, y);
//        crv->SetMousePosition(x, y);
//        (*crv)(view::CallRenderView::CALL_SETCURSOR2DPOSITION);
//    }
//}
//
//
///*
// * view::AbstractOverrideView::SetInputModifier
// */
//void view::AbstractOverrideView::SetInputModifier(view::Modifier mod, bool down) {
//    view::CallRenderView *crv = this->getCallRenderView();
//    if (crv != NULL) {
//        crv->SetInputModifier(mod, down);
//        (*crv)(view::CallRenderView::CALL_SETINPUTMODIFIER);
//    }
//}


/*
 * view::AbstractOverrideView::UpdateFreeze
 */
void view::AbstractOverrideView::UpdateFreeze(bool freeze) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        (*crv)(freeze
            ? view::CallRenderView::CALL_FREEZE
            : view::CallRenderView::CALL_UNFREEZE);
    }
}


bool view::AbstractOverrideView::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderView::FnOnKey)) return false;

    return true;
}


bool view::AbstractOverrideView::OnChar(unsigned int codePoint) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderView::FnOnChar)) return false;

    return true;
}


bool view::AbstractOverrideView::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseButton;
    evt.mouseButtonData.button = button;
    evt.mouseButtonData.action = action;
    evt.mouseButtonData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderView::FnOnMouseButton)) return false;

    return true;
}


bool view::AbstractOverrideView::OnMouseMove(double x, double y) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseMove;
    evt.mouseMoveData.x = x;
    evt.mouseMoveData.y = y;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderView::FnOnMouseMove)) return false;

    return true;
}


bool view::AbstractOverrideView::OnMouseScroll(double dx, double dy) {
    auto* cr = this->getCallRenderView();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRenderView::FnOnMouseScroll)) return false;

    return true;
}


/*
 * view::AbstractOverrideView::disconnectOutgoingRenderCall
 */
void view::AbstractOverrideView::disconnectOutgoingRenderCall(void) {
    this->renderViewSlot.ConnectCall(NULL);
}


/*
 * view::AbstractOverrideView::getConnectedView
 */
view::AbstractView *view::AbstractOverrideView::getConnectedView(void) const {
    Call* c = const_cast<CallerSlot*>(&this->renderViewSlot)->CallAs<Call>();
    if ((c == NULL) || (c->PeekCalleeSlot() == NULL)) return NULL;
    return const_cast<view::AbstractView*>(
        dynamic_cast<const view::AbstractView*>(c->PeekCalleeSlot()->Parent().get()));
}


/*
 * view::AbstractOverrideView::packMouseCoordinates
 */
void view::AbstractOverrideView::packMouseCoordinates(float &x, float &y) {
    // intentionally empty
    // do something smart in the derived classes
}
