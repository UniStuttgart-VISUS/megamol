/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/view/View3DGL.h"

#include "GlobalValueStore.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"

using namespace megamol::mmstd_gl;
using namespace megamol::mmstd_gl::view;

/*
 * View3DGL::View3DGL
 */
View3DGL::View3DGL() {
    this->_rhsRenderSlot.SetCompatibleCall<CallRender3DGLDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);
    // Override renderSlot behavior
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar), &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove), &AbstractView::OnMouseMoveCallback);
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::AbstractCallRender::FunctionName(core::view::AbstractCallRender::FnRender),
        &AbstractView::OnRenderView);
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::AbstractCallRender::FunctionName(core::view::AbstractCallRender::FnGetExtents),
        &AbstractView::GetExtents);
    // CallRenderViewGL
    this->_lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        CallRenderViewGL::FunctionName(CallRenderViewGL::CALL_RESETVIEW), &AbstractView::OnResetView);
    this->MakeSlotAvailable(&this->_lhsRenderSlot);

    this->_rhsRenderSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);
}

/*
 * View3DGL::~View3DGL
 */
View3DGL::~View3DGL() {
    this->Release();
}

megamol::frontend_resources::ImageWrapper View3DGL::Render(double time, double instanceTime) {

    BaseView::beforeRender(time, instanceTime);

    // clear fbo before sending it down the rendering call
    // the view is the owner of this fbo and therefore responsible
    // for clearing it at the beginning of a render frame
    _fbo->bind();
    auto bgcol = this->BackgroundColor();
    glClearColor(bgcol.r * bgcol.a, bgcol.g * bgcol.a, bgcol.b * bgcol.a, bgcol.a); // Premultiply alpha
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    CallRender3DGL* cr3d = this->_rhsRenderSlot.CallAs<CallRender3DGL>();

    if (cr3d != NULL) {
        // set camera and fbo in rendering call
        cr3d->SetViewResolution({_fbo->getWidth(), _fbo->getHeight()});
        cr3d->SetFramebuffer(_fbo);
        cr3d->SetCamera(this->_camera);

        // call the rendering call
        (*cr3d)(CallRender3DGL::FnRender);
    }

    BaseView::afterRender();

    return GetRenderingResult();
}

megamol::frontend_resources::ImageWrapper View3DGL::GetRenderingResult() const {
    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib_gl::graphics::gl::FramebufferObject seems to use RGBA8
    unsigned int fbo_color_buffer_gl_handle =
        _fbo->getColorAttachment(0)->getName(); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
    size_t fbo_width = _fbo->getWidth();
    size_t fbo_height = _fbo->getHeight();

    return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
}

void View3DGL::Resize(unsigned int width, unsigned int height) {

    BaseView::Resize(width, height);

    bool create_fbo = false;
    if (_fbo == nullptr) {
        create_fbo = true;
    } else if (((_fbo->getWidth() != width) || (_fbo->getHeight() != height)) && width != 0 && height != 0) {
        create_fbo = true;
    }

    if (create_fbo) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
        try {
            _fbo = std::make_shared<glowl::FramebufferObject>(width, height);
            _fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
            // TODO: check completness and throw if not?
        } catch (glowl::FramebufferObjectException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[View3DGL] Unable to create framebuffer object: %s\n", exc.what());
        }
    }
}

/*
 * View3DGL::create
 */
bool View3DGL::create() {
    // intialize fbo with dummy size until the actual size is set during first call to Resize
    this->_fbo = std::make_shared<glowl::FramebufferObject>(1, 1);

    const auto arcball_key = "arcball";

    // new frontend has global key-value resource
    auto maybe = this->frontend_resources.get<megamol::frontend_resources::GlobalValueStore>().maybe_get(arcball_key);
    if (maybe.has_value()) {
        this->_camera_controller.setArcballDefault(vislib::CharTraitsA::ParseBool(maybe.value().c_str()));
    }

    this->_firstImg = true;

    return true;
}
