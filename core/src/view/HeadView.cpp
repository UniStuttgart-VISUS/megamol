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
tickSlot("tick", "Connects to a module that needs a tick") {

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
view::ImageWrapper view::HeadView::Render(double time, double instanceTime, bool present_fbo) {
    CallRenderViewGL* view = this->viewSlot.CallAs<CallRenderViewGL>();

    if (view != nullptr) {
        std::unique_ptr<CallRenderViewGL> last_view_call = nullptr;

        if (this->override_view_call != nullptr) {
            last_view_call = std::make_unique<CallRenderViewGL>(*view);
            *view = *this->override_view_call;
        }

        view->SetInstanceTime(instanceTime);
        view->SetTime(static_cast<float>(time));

        if (this->doHookCode()) {
            this->doBeforeRenderHook();
        }

        (*view)(CallRenderViewGL::CALL_RENDER);
        auto fbo = view->GetFramebuffer();

        if (this->doHookCode()) {
            this->doAfterRenderHook();
        }

        if (last_view_call != nullptr) {
            *view = *last_view_call;
        }

        ImageWrapper::DataChannels channels =
            ImageWrapper::DataChannels::RGBA8; // vislib::graphics::gl::FramebufferObject seems to use RGBA8
        unsigned int fbo_color_buffer_gl_handle =
            fbo->getColorAttachment(0)->getTextureHandle(); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
        size_t fbo_width = fbo->getWidth();
        size_t fbo_height = fbo->getHeight();

        return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
    }

    auto* tick = this->tickSlot.CallAs<job::TickCall>();

    if (tick != nullptr) {
        (*tick)(0);
    }

    return GetRenderingResult();
}

view::ImageWrapper megamol::core::view::HeadView::GetRenderingResult() const {
    return frontend_resources::wrap_image<WrappedImageType::ByteArray>(
        {0, 0}, nullptr, ImageWrapper::DataChannels::RGBA8);
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

    double time = view->Time();
    double instanceTime = view->InstanceTime();

    this->Render(time, instanceTime, false);

    this->override_view_call = nullptr;

    return true;
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
