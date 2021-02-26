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

    this->render_view_slot.SetCompatibleCall<core::view::CallRenderViewGLDescription>();
    this->MakeSlotAvailable(&this->render_view_slot);
    this->MakeSlotAvailable(&this->_lhsRenderSlot);
}


GUIView::~GUIView() {
    this->Release();
}


bool GUIView::create() {

    if (this->_fbo == nullptr) {
        this->_fbo = std::make_shared<vislib::graphics::gl::FramebufferObject>();
    }

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

void GUIView::Render(const mmcRenderViewContext& context, core::Call* call) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }
    if (crv) {
        // Camera
        core::view::Camera_2 cam;
        crv->GetCamera(cam);
        cam_type::snapshot_type snapshot;
        cam_type::matrix_type viewTemp, projTemp;
        cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);

        auto viewport_rect = cam.resolution_gate();
        auto viewport =
            glm::vec2(static_cast<float>(viewport_rect.width()), static_cast<float>(viewport_rect.height()));

        if (this->_fbo->IsValid()) {
            if ((this->_fbo->GetWidth() != viewport.x) || (this->_fbo->GetHeight() != viewport.y)) {
                this->_fbo->Release();
                if (!this->_fbo->Create(viewport.x, viewport.y, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                        vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE)) {
                    throw vislib::Exception(
                        "[TILEVIEW] Unable to create image framebuffer object.", __FILE__, __LINE__);
                    return;
                }
            }
        }

        crv->SetFramebufferObject(_fbo);
        crv->SetInstanceTime(context.InstanceTime);
        // Should be negative to trigger animation! (see View3DGL.cpp line ~612 | View2DGL.cpp line ~661):
        crv->SetTime(-1.0f);
        this->gui.PreDraw(viewport, viewport, crv->InstanceTime());
        (*crv)(core::view::AbstractCallRender::FnRender);
        this->gui.PostDraw();
    } else {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (this->overrideCall != nullptr) {
            auto override_cam = this->overrideCall->GetCamera();
            cam_type::snapshot_type override_snapshot;
            cam_type::matrix_type override_viewTemp, override_projTemp;
            override_cam.calc_matrices(
                override_snapshot, override_viewTemp, override_projTemp, core::thecam::snapshot_content::all);
            auto viewport_rect = override_cam.resolution_gate();
            auto viewport =
                glm::vec2(static_cast<float>(viewport_rect.width()), static_cast<float>(viewport_rect.height()));
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
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
    if (crv) {
        (*crv)(core::view::CallRenderViewGL::CALL_RESETVIEW);
    }
}


void GUIView::Resize(unsigned int width, unsigned int height) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
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
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
    if (crv) {
        auto callType =
            freeze ? core::view::CallRenderViewGL::CALL_FREEZE : core::view::CallRenderViewGL::CALL_UNFREEZE;
        (*crv)(callType);
    }
}


bool GUIView::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    bool input_consumed = this->gui.OnKey(key, action, mods);

    if (!input_consumed) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
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

    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
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
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
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
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
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
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
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
    megamol::core::view::CallRenderViewGL* crv = dynamic_cast<megamol::core::view::CallRenderViewGL*>(&call);
    if (crv == nullptr)
        return false;

    this->overrideCall = crv;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = crv->Time();
    context.InstanceTime = crv->InstanceTime();
    // XXX Affinity?

    this->Render(context, &call);

    this->overrideCall = nullptr;

    return true;
}
