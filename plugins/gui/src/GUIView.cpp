/*
 * GUIView.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIView.h"


using namespace megamol;
using namespace megamol::gui;


GUIView::GUIView()
        : core::view::AbstractView()
        , overrideCall(nullptr)
        , render_view_slot("renderview", "Connects to a preceding RenderView that will be decorated with a GUI")
        , gui() {

    this->render_view_slot.SetCompatibleCall<core::view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render_view_slot);
}


GUIView::~GUIView() {
    this->Release();
}


bool GUIView::create() {

    if (this->GetCoreInstance()->IsmmconsoleFrontendCompatible()) {
        return gui.CreateContext(megamol::gui::GUIImGuiAPI::OPEN_GL, this->GetCoreInstance());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] GUIView module can only be used with mmconsole frontend. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }
}


void GUIView::release() {}


void GUIView::unpackMouseCoordinates(float& x, float& y) {
    GLint vpw = 1;
    GLint vph = 1;
    if (this->overrideCall == nullptr) {
        GLint vp[4];
        ::glGetIntegerv(GL_VIEWPORT, vp);
        vpw = vp[2];
        vph = vp[3];
    } else {
        vpw = this->overrideCall->ViewportWidth();
        vph = this->overrideCall->ViewportHeight();
    }
    x *= static_cast<float>(vpw);
    y *= static_cast<float>(vph);
}


float GUIView::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}


unsigned int GUIView::GetCameraSyncNumber(void) const {
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Unsupported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return 0u;
}


void GUIView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Unsupported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
}


void GUIView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Unsupported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
}


void GUIView::Render(const mmcRenderViewContext& context) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }
    if (crv) {
        crv->SetOutputBuffer(GL_BACK);
        crv->SetInstanceTime(context.InstanceTime);
        // Should be negative to trigger animation! (see View3D_2.cpp line ~612 | View2D.cpp line ~661):
        crv->SetTime(-1.0f);
        auto viewport_rect = crv->GetViewport();
        auto viewport =
            glm::vec2(static_cast<float>(viewport_rect.Width()), static_cast<float>(viewport_rect.Height()));
        this->gui.PreDraw(viewport, viewport, crv->InstanceTime());
        (*crv)(core::view::AbstractCallRender::FnRender);
        this->gui.PostDraw();
    } else {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (this->overrideCall != nullptr) {
            auto viewport_rect = this->overrideCall->GetViewport();
            auto viewport =
                glm::vec2(static_cast<float>(viewport_rect.Width()), static_cast<float>(viewport_rect.Height()));
            this->gui.PreDraw(viewport, viewport, context.InstanceTime);
            this->gui.PostDraw();
        } else {
            GLint vp[4];
            glGetIntegerv(GL_VIEWPORT, vp);
            auto viewport = glm::vec2(static_cast<float>(vp[2]), static_cast<float>(vp[3]));
            this->gui.PreDraw(viewport, viewport, context.InstanceTime);
            this->gui.PostDraw();
        }
    }
    if (this->doHookCode()) {
        this->doAfterRenderHook();
    }

    this->gui.SynchronizeGraphs();
}


void GUIView::ResetView(void) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        (*crv)(core::view::CallRenderView::CALL_RESETVIEW);
    }
}


void GUIView::Resize(unsigned int width, unsigned int height) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        // der ganz ganz dicke "because-i-know"-Knueppel
        AbstractView* view = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
        if (view != nullptr) {
            view->Resize(width, height);
        }
    }
}


void GUIView::UpdateFreeze(bool freeze) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        auto callType = freeze ? core::view::CallRenderView::CALL_FREEZE : core::view::CallRenderView::CALL_UNFREEZE;
        (*crv)(callType);
    }
}


bool GUIView::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    bool input_consumed = this->gui.OnKey(key, action, mods);

    if (!input_consumed) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr)
            return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        crv->SetInputEvent(evt);
        return (*crv)(core::view::InputCall::FnOnKey);
    }

    return true;
}


bool GUIView::OnChar(unsigned int codePoint) {

    this->gui.OnChar(codePoint);

    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        crv->SetInputEvent(evt);
        return (*crv)(core::view::InputCall::FnOnChar);
    }

    return true;
}


bool GUIView::OnMouseMove(double x, double y) {

    bool input_consumed = this->gui.OnMouseMove(x, y);

    if (!input_consumed) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr)
            return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        crv->SetInputEvent(evt);
        return (*crv)(core::view::InputCall::FnOnMouseMove);
    }

    return true;
}


bool GUIView::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    bool input_consumed = this->gui.OnMouseButton(button, action, mods);

    if (!input_consumed) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr)
            return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        crv->SetInputEvent(evt);
        return (*crv)(core::view::InputCall::FnOnMouseButton);
    }

    return true;
}


bool GUIView::OnMouseScroll(double dx, double dy) {

    bool input_consumed = this->gui.OnMouseScroll(dx, dy);

    if (!input_consumed) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr)
            return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        crv->SetInputEvent(evt);
        return (*crv)(core::view::InputCall::FnOnMouseScroll);
    }

    return true;
}


bool GUIView::OnRenderView(megamol::core::Call& call) {
    megamol::core::view::CallRenderView* crv = dynamic_cast<megamol::core::view::CallRenderView*>(&call);
    if (crv == nullptr)
        return false;

    this->overrideCall = crv;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = crv->Time();
    context.InstanceTime = crv->InstanceTime();
    // XXX Affinity?

    this->Render(context);

    this->overrideCall = nullptr;

    return true;
}
