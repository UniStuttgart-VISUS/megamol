/*
 * HeadView.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/job/TickCall.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/HeadView.h"

#include "vislib/sys/Log.h"
#include "vislib/Trace.h"

#include <memory>

using namespace megamol;
using namespace megamol::core;
using vislib::sys::Log;

/*
 * view::HeadView::HeadView
 */
view::HeadView::HeadView(void) : AbstractView(),
viewSlot("view", "Connects to a view"),
tickSlot("tick", "Connects to a module that needs a tick"),
override_view_call(nullptr) {

    this->viewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
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
void view::HeadView::Render(const mmcRenderViewContext& context) {
    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) {
        std::unique_ptr<CallRenderView> last_view_call = nullptr;

        if (this->override_view_call != nullptr) {
            last_view_call = std::make_unique<CallRenderView>(*view);
            *view = *this->override_view_call;
        }
        else {
            const_cast<vislib::math::Rectangle<int>&>(view->GetViewport()).Set(0, 0, this->width, this->height);
        }

        view->SetInstanceTime(context.InstanceTime);
        view->SetTime(static_cast<float>(context.Time));

        if (this->doHookCode()) {
            this->doBeforeRenderHook();
        }

        (*view)(CallRenderView::CALL_RENDER);

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
    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) (*view)(CallRenderView::CALL_RESETVIEW);
}


/*
 * view::HeadView::Resize
 */
void view::HeadView::Resize(unsigned int width, unsigned int height) {
    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

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
    view::CallRenderView *view = dynamic_cast<view::CallRenderView *>(&call);
    if (view == nullptr) return false;

    this->override_view_call = view;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));

    context.Time = view->Time();
    context.InstanceTime = view->InstanceTime();

    this->Render(context);

    this->override_view_call = nullptr;

    return true;
}


/*
 * view::HeadView::UpdateFreeze
 */
void view::HeadView::UpdateFreeze(bool freeze) {
    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) (*view)(freeze ? CallRenderView::CALL_FREEZE : CallRenderView::CALL_UNFREEZE);
}


bool view::HeadView::OnKey(Key key, KeyAction action, Modifiers mods) {

    bool consumed = false;

    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();
    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderView::FnOnKey)) consumed = true;
    }

    return consumed;
}


bool view::HeadView::OnChar(unsigned int codePoint) {

    bool consumed = false;

    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderView::FnOnChar)) consumed = true;
    }

    return consumed;
}


bool view::HeadView::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {

    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderView::FnOnMouseButton)) return true;
    }

    return true;
}


bool view::HeadView::OnMouseMove(double x, double y) {

    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderView::FnOnMouseMove)) return true;
    }

    return true;
}


bool view::HeadView::OnMouseScroll(double dx, double dy) {

    CallRenderView *view = this->viewSlot.CallAs<CallRenderView>();

    if (view != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;

        view->SetInputEvent(evt);

        if ((*view)(view::CallRenderView::FnOnMouseScroll)) return true;
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
