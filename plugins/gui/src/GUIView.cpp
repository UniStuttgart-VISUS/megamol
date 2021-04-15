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
        this->_fbo = std::make_shared<glowl::FramebufferObject>(1,1);
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
    GLint vpw = _fbo->getWidth();
    GLint vph = _fbo->getHeight();

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

core::view::ImageWrapper GUIView::Render(double time, double instanceTime, bool present_fbo) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }
    if (crv) {
        // Camera
        auto viewport =
            glm::vec2(static_cast<float>(_fbo->getWidth()), static_cast<float>(_fbo->getHeight()));

        crv->SetFramebufferObject(_fbo);
        crv->SetInstanceTime(instanceTime);
        // Should be negative to trigger animation! (see View3DGL.cpp line ~612 | View2DGL.cpp line ~661):
        crv->SetTime(-1.0f);
        this->gui.PreDraw(viewport, viewport, crv->InstanceTime());
        (*crv)(core::view::AbstractCallRender::FnRender);
        this->gui.PostDraw();
    } else {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        auto viewport = glm::vec2(static_cast<float>(_fbo->getWidth()), static_cast<float>(_fbo->getHeight()));
        this->gui.PreDraw(viewport, viewport, instanceTime);
        this->gui.PostDraw();
    }
    if (this->doHookCode()) {
        this->doAfterRenderHook();
    }

    this->gui.SynchronizeGraphs();

    return GetRenderingResult();
}

core::view::ImageWrapper megamol::gui::GUIView::GetRenderingResult() const {

    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib::graphics::gl::FramebufferObject seems to use RGBA8
    unsigned int fbo_color_buffer_gl_handle =
        _fbo->getColorAttachment(0)->getTextureHandle(); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
    size_t fbo_width = _fbo->getWidth();
    size_t fbo_height = _fbo->getHeight();

    return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
}


void GUIView::ResetView(void) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderViewGL>();
    if (crv) {
        (*crv)(core::view::CallRenderViewGL::CALL_RESETVIEW);
    }
}


void GUIView::Resize(unsigned int width, unsigned int height) {
    if ((this->_fbo->getWidth() != width) || (this->_fbo->getHeight() != height)) {

        glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
        try {
            _fbo = std::make_shared<glowl::FramebufferObject>(width, height);
            _fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

            // TODO: check completness and throw if not?
        } catch (glowl::FramebufferObjectException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[View2DGL] Unable to create image framebuffer object: %s\n", exc.what());
        }
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

    double time = crv->Time();
    double instanceTime = crv->InstanceTime();

    _camera = crv->GetCamera();
    this->Resize(crv->GetFramebufferObject()->getWidth(), crv->GetFramebufferObject()->getHeight());

    this->Render(time, instanceTime, false);

    return true;
}
