/*
 * HeadView.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/job/TickCall.h"
#include "mmcore/view/CallRenderViewGL.h"
#include "mmcore/view/HeadView.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/Trace.h"

#include <memory>

using namespace megamol;
using namespace megamol::core;
using megamol::core::utility::log::Log;

/*
 * view::HeadView::HeadView
 */
view::HeadView::HeadView(void) : AbstractView(),
viewSlot("view", "Connects to a view"),
tickSlot("tick", "Connects to a module that needs a tick"),
override_view_call(nullptr) {

    this->viewSlot.SetCompatibleCall<view::CallRenderViewGLDescription>();
    this->MakeSlotAvailable(&this->viewSlot);

    this->tickSlot.SetCompatibleCall<job::TickCall::TickCallDescription>();
    this->MakeSlotAvailable(&this->tickSlot);
}


/*
 * view::HeadView::~HeadView
 */
view::HeadView::~HeadView(void) {
    this->Release();
}


/*
 * view::HeadView::DefaultTime
 */
float view::HeadView::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}


/*
 * view::HeadView::GetCameraSyncNumber
 */
unsigned int view::HeadView::GetCameraSyncNumber(void) const {
    Log::DefaultLog.WriteWarn("HeadView::GetCameraSyncNumber unsupported");
    return 0u;
}


/*
 * view::HeadView::SerialiseCamera
 */
void view::HeadView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    Log::DefaultLog.WriteWarn("HeadView::SerialiseCamera unsupported");
}


/*
 * view::HeadView::DeserialiseCamera
 */
void view::HeadView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    Log::DefaultLog.WriteWarn("HeadView::DeserialiseCamera unsupported");
}


/*
 * view::HeadView::Render
 */
void view::HeadView::Render(const mmcRenderViewContext& context, Call* call) {
    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    auto cam  = view->GetCamera();

    if (view != nullptr) {
        std::unique_ptr<CallRenderViewGL> last_view_call = nullptr;

        if (this->override_view_call != nullptr) {
            last_view_call = std::make_unique<CallRenderViewGL>(*view);
            *view = *this->override_view_call;
        }
        else {
            //const_cast<vislib::math::Rectangle<int>&>(view->GetViewport()).Set(0, 0, this->width, this->height);
            thecam::math::rectangle<int> rect;
            rect.bottom() = 0;
            rect.left() = 0;
            rect.right() = this->width;
            rect.top() = this->height;
            cam.image_tile.operator()(rect);
        }

        view->SetInstanceTime(context.InstanceTime);
        view->SetTime(static_cast<float>(context.Time));

        if (this->doHookCode()) {
            this->doBeforeRenderHook();
        }

        (*view)(CallRenderViewGL::CALL_RENDER);

        if (this->doHookCode()) {
            this->doAfterRenderHook();
        }

        if (last_view_call != nullptr) {
            *view = *last_view_call;
        }
    }

    auto* tick = this->tickSlot.CallAs<job::TickCall>();

    if (tick != nullptr)
    {
        (*tick)(0);
    }
}


/*
 * view::HeadView::ResetView
 */
void view::HeadView::ResetView(void) {
    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) (*view)(CallRenderViewGL::CALL_RESETVIEW);
}


/*
 * view::HeadView::Resize
 */
void view::HeadView::Resize(unsigned int width, unsigned int height) {
    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    this->width = width;
    this->height = height;

    if (view != nullptr) {
        AbstractView *abstract_view = const_cast<AbstractView*>(dynamic_cast<const AbstractView *>(static_cast<const Module*>(view->PeekCalleeSlot()->Owner())));

        if (abstract_view != nullptr) {
            abstract_view->Resize(width, height);
        }
    }
}


/*
 * view::HeadView::OnRenderView
 */
bool view::HeadView::OnRenderView(Call& call) {
    view::CallRenderViewGL *view = dynamic_cast<view::CallRenderViewGL *>(&call);
    if (view == nullptr) return false;

    this->override_view_call = view;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));

    context.Time = view->Time();
    context.InstanceTime = view->InstanceTime();

    this->Render(context, &call);

    this->override_view_call = nullptr;

    return true;
}


/*
 * view::HeadView::UpdateFreeze
 */
void view::HeadView::UpdateFreeze(bool freeze) {
    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) (*view)(freeze ? CallRenderViewGL::CALL_FREEZE : CallRenderViewGL::CALL_UNFREEZE);
}


bool view::HeadView::OnKey(Key key, KeyAction action, Modifiers mods) {

    bool consumed = false;

    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();
    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderViewGL::FnOnKey)) consumed = true;
    }

    return consumed;
}


bool view::HeadView::OnChar(unsigned int codePoint) {

    bool consumed = false;

    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderViewGL::FnOnChar)) consumed = true;
    }

    return consumed;
}


bool view::HeadView::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {

    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderViewGL::FnOnMouseButton)) return true;
    }

    return true;
}


bool view::HeadView::OnMouseMove(double x, double y) {

    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderViewGL::FnOnMouseMove)) return true;
    }

    return true;
}


bool view::HeadView::OnMouseScroll(double dx, double dy) {

    CallRenderViewGL *view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderViewGL::FnOnMouseScroll)) return true;
    }

    return true;
}


/*
 * view::HeadView::create
 */
bool view::HeadView::create(void) {
    // nothing to do
    return true;
}


/*
 * view::HeadView::release
 */
void view::HeadView::release(void) {
}


/*
 * view::HeadView::unpackMouseCoordinates
 */
void view::HeadView::unpackMouseCoordinates(float &x, float &y) {
}
