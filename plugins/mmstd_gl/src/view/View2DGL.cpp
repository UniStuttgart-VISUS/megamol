/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/view/View2DGL.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "vislib/Trace.h"
#include "vislib/math/Matrix4.h"
#include "vislib/math/Rectangle.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"


using namespace megamol::mmstd_gl;


/*
 * view::View2DGL::View2DGL
 */
view::View2DGL::View2DGL(void) {

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

    this->_rhsRenderSlot.SetCompatibleCall<CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);
}


/*
 * view::View2DGL::~View2DGL
 */
view::View2DGL::~View2DGL(void) {
    this->Release();
}

/*
 * view::View2DGL::Render
 */
megamol::core::view::ImageWrapper view::View2DGL::Render(double time, double instanceTime) {

    AbstractView::beforeRender(time, instanceTime);

    // clear fbo before sending it down the rendering call
    // the view is the owner of this fbo and therefore responsible
    // for clearing it at the beginning of a render frame
    this->_fbo->bind();
    auto bgcol = this->BackgroundColor();
    glClearColor(bgcol.r * bgcol.a, bgcol.g * bgcol.a, bgcol.b * bgcol.a, bgcol.a); // Premultiply alpha
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();

    if (cr2d != NULL) {
        cr2d->SetViewResolution({_fbo->getWidth(), _fbo->getHeight()});
        cr2d->SetFramebuffer(_fbo);
        cr2d->SetCamera(_camera);

        (*cr2d)(core::view::AbstractCallRender::FnRender);
    }

    AbstractView::afterRender();

    return GetRenderingResult();
}

megamol::core::view::ImageWrapper view::View2DGL::GetRenderingResult() const {
    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib_gl::graphics::gl::FramebufferObject seems to use RGBA8
    unsigned int fbo_color_buffer_gl_handle =
        _fbo->getColorAttachment(0)->getName(); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
    size_t fbo_width = _fbo->getWidth();
    size_t fbo_height = _fbo->getHeight();

    return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
}

/*
 * view::View2DGL::create
 */
bool view::View2DGL::create(void) {

    this->_firstImg = true;

    // intialize fbo with dummy size until the actual size is set during first call to Resize
    this->_fbo = std::make_shared<glowl::FramebufferObject>(1, 1);

    return true;
}


void view::View2DGL::Resize(unsigned int width, unsigned int height) {
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
                "[View2DGL] Unable to create framebuffer object: %s\n", exc.what());
        }
    }
}
