/*
 * ContextToGL.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/ContextToGL.h"

namespace megamol::core::view {





ContextToGL::ContextToGL(void)
    : Renderer3DModuleGL()
    , _getContextSlot("getContext", "Slot for non-GL context")
{

    this->_getContextSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->_getContextSlot);

}

ContextToGL::~ContextToGL(void) {
    this->Release();
}

bool ContextToGL::create(void) {
    return true;
}

void ContextToGL::release(void) {
}

bool ContextToGL::GetExtents(CallRender3DGL& call) {

    auto cr = _getContextSlot.CallAs<CallRender3D>();
    if (cr == nullptr) return false;
    // no copy constructor available
    cr->SetTimeFramesCount(call.TimeFramesCount());
    cr->SetTime(call.Time());
    cr->SetInstanceTime(call.InstanceTime());
    cr->SetLastFrameTime(call.LastFrameTime());
    cr->SetBackgroundColor(call.BackgroundColor());
    auto cpy_cam = call.GetCamera();
    cr->SetCameraState(cpy_cam);
    cr->AccessBoundingBoxes() = call.AccessBoundingBoxes();

    if (!_framebuffer) {
        _framebuffer = std::make_shared<CPUFramebuffer>();
    }
    cr->setGenericFramebuffer(_framebuffer);

    (*cr)(view::CallRender3D::FnGetExtents);

    return true;
}

bool ContextToGL::Render(CallRender3DGL& call) {

    auto cr = _getContextSlot.CallAs<CallRender3D>();
    if (cr == nullptr) return false;
    // no copy constructor available
    cr->SetTimeFramesCount(call.TimeFramesCount());
    cr->SetTime(call.Time());
    cr->SetInstanceTime(call.InstanceTime());
    cr->SetLastFrameTime(call.LastFrameTime());
    cr->SetBackgroundColor(call.BackgroundColor());
    auto cpy_cam = call.GetCamera();
    cr->SetCameraState(cpy_cam);
    cr->AccessBoundingBoxes() = call.AccessBoundingBoxes();

    if (!_framebuffer) {
        _framebuffer = std::make_shared<CPUFramebuffer>();
    }
    cr->setGenericFramebuffer(_framebuffer);

    (*cr)(view::CallRender3D::FnRender);

    Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);

    auto width = cam.resolution_gate().width();
    auto height = cam.resolution_gate().height();

    auto fbo = call.FrameBufferObject();
    if (fbo != NULL) {

        if (fbo->IsValid()) {
            if ((fbo->GetWidth() != width) || (fbo->GetHeight() != height)) {
                fbo->Release();
            }
        }
        if (!fbo->IsValid()) {
            fbo->Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
        }
        if (fbo->IsValid() && !fbo->IsEnabled()) {
            fbo->Enable();
        }

        fbo->BindColourTexture();
        glClear(GL_COLOR_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, _framebuffer->colorBuffer.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        fbo->BindDepthTexture();
        glClear(GL_DEPTH_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, _framebuffer->depthBuffer.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        if (fbo->IsValid()) {
            fbo->Disable();
            // fbo->DrawColourTexture();
            // fbo->DrawDepthTexture();
        }
    } else {
        return false;
    }
    return true;
}

} // namespace megamol::core::view
