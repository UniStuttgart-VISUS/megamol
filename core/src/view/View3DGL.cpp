/*
 * View3DGL.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3DGL.h"


#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRender3DGL.h"
#include "mmcore/view/CallRenderViewGL.h"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3DGL::View3DGL
 */
View3DGL::View3DGL(void) : view::AbstractView3D<glowl::FramebufferObject, gl3D_fbo_create_or_resize, Camera3DController, Camera3DParameters>() {
    this->_rhsRenderSlot.SetCompatibleCall<CallRender3DGLDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);
    // Override renderSlot behavior
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnChar),
        &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseButton), &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseMove), &AbstractView::OnMouseMoveCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseScroll), &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &AbstractView::OnRenderView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &AbstractView::GetExtents);
    // CallRenderViewGL
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_RESETVIEW), &AbstractView::OnResetView);
    this->MakeSlotAvailable(&this->_lhsRenderSlot);

    this->_rhsRenderSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);
}

/*
 * View3DGL::~View3DGL
 */
View3DGL::~View3DGL(void) {
    this->Release();
}

ImageWrapper megamol::core::view::View3DGL::Render(double time, double instanceTime, bool present_fbo) {
    CallRender3DGL* cr3d = this->_rhsRenderSlot.CallAs<CallRender3DGL>();

    if (cr3d != NULL) {

        AbstractView3D::beforeRender(time, instanceTime);

        // clear fbo before sending it down the rendering call
        // the view is the owner of this fbo and therefore responsible
        // for clearing it at the beginning of a render frame
        _fbo->bind();
        auto bgcol = this->BkgndColour();
        glClearColor(bgcol.r, bgcol.g, bgcol.b, bgcol.a);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // set camera and fbo in rendering call
        cr3d->SetFramebufferObject(_fbo);
        cr3d->SetCamera(this->_camera);

        // call the rendering call
        (*cr3d)(view::CallRender3DGL::FnRender);

        AbstractView3D::afterRender();
    }

    if (present_fbo) {
        // Blit the final image to the default framebuffer of the window.
        // Technically, the view's fbo should always match the size of the window so a blit is fine.
        // Eventually, presenting the fbo will become the frontends job.
        // Bind and blit framebuffer.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        _fbo->bindToRead(0);
        glBlitFramebuffer(0, 0, _fbo->getWidth(), _fbo->getHeight(), 0, 0, _fbo->getWidth(), _fbo->getHeight(),
            GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    }

    return GetRenderingResult();
}

ImageWrapper megamol::core::view::View3DGL::GetRenderingResult() const {
    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib::graphics::gl::FramebufferObject seems to use RGBA8
    unsigned int fbo_color_buffer_gl_handle =
        _fbo->getColorAttachment(0)->getTextureHandle(); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
    size_t fbo_width = _fbo->getWidth();
    size_t fbo_height = _fbo->getHeight();

    return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
}

void megamol::core::view::View3DGL::ResetView() {
    AbstractView3D::ResetView(static_cast<float>(_fbo->getWidth())/static_cast<float>(_fbo->getHeight()));
}

bool megamol::core::view::View3DGL::OnRenderView(Call& call) {
    view::CallRenderViewGL* crv = dynamic_cast<view::CallRenderViewGL*>(&call);
    if (crv == NULL) {
        return false;
    }

    // get time from incoming call
    double time = crv->Time();
    if (time < 0.0f)
        time = this->DefaultTime(crv->InstanceTime());
    double instanceTime = crv->InstanceTime();

    auto fbo = _fbo;
    _fbo = crv->GetFramebufferObject();

    auto cam_cpy = _camera;
    auto cam_pose = _camera.get<Camera::Pose>();
    auto cam_type = _camera.get<Camera::ProjectionType>();
    if (cam_type == Camera::ORTHOGRAPHIC) {
        auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();
        cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
        _camera = Camera(cam_pose, cam_intrinsics);
    } else if (cam_type == Camera::ORTHOGRAPHIC) {
        auto cam_intrinsics = _camera.get<Camera::PerspectiveParameters>();
        cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
        _camera = Camera(cam_pose, cam_intrinsics);
    }
    
    this->Render(time, instanceTime, false);

    _fbo = fbo;
    _camera = cam_cpy;

    return true;
}

/*
 * View3DGL::create
 */
bool View3DGL::create(void) {

    AbstractView3D::create();

    // intialize fbo with dummy size until the actual size is set during first call to Resize
    this->_fbo = std::make_shared<glowl::FramebufferObject>(1,1);

    return true;
}
