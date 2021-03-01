/*
 * ContextToGL.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/ContextToGL.h"
#include "mmcore/utility/log/Log.h"

namespace megamol::core::view {

using utility::log::Log;

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
    if (!_utils.isInitialized()) {
        if (!_utils.InitPrimitiveRendering(this->GetCoreInstance()->ShaderSourceFactory())) {
            Log::DefaultLog.WriteError("[ContextToGL] Unable to initialize RenderUtility.");
        }
    }

    return true;
}

void ContextToGL::release(void) {
}

bool ContextToGL::GetExtents(CallRender3DGL& call) {

    auto cr = _getContextSlot.CallAs<CallRender3D>();
    if (cr == nullptr) return false;
    // no copy constructor available
    auto cast_in = dynamic_cast<AbstractCallRender*>(&call);
    auto cast_out = dynamic_cast<AbstractCallRender*>(cr);
    *cast_out = *cast_in;

    if (!_framebuffer) {
        _framebuffer = std::make_shared<CPUFramebuffer>();
    }
    cr->SetFramebuffer(_framebuffer);

    (*cr)(view::CallRender3D::FnGetExtents);

    call.AccessBoundingBoxes() = cr->AccessBoundingBoxes();
    call.SetTimeFramesCount(cr->TimeFramesCount());

    return true;
}

bool ContextToGL::Render(CallRender3DGL& call) {

    auto cr = _getContextSlot.CallAs<CallRender3D>();
    if (cr == nullptr) return false;
    // no copy constructor available
    auto cast_in = dynamic_cast<AbstractCallRender*>(&call);
    auto cast_out = dynamic_cast<AbstractCallRender*>(cr);
    *cast_out = *cast_in;

    if (!_framebuffer) {
        _framebuffer = std::make_shared<CPUFramebuffer>();
    }
    cr->SetFramebuffer(_framebuffer);

    (*cr)(view::CallRender3D::FnRender);

    Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);

    auto width = cam.resolution_gate().width();
    auto height = cam.resolution_gate().height();

    auto lhs_fbo = call.GetFramebufferObject();
    if (lhs_fbo != NULL) {

        // module own fbo
        auto new_fbo = vislib::graphics::gl::FramebufferObject();
        new_fbo.Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
            vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
        new_fbo.Enable();

        new_fbo.BindColourTexture();
        glClear(GL_COLOR_BUFFER_BIT);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, _framebuffer->colorBuffer.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        if (_framebuffer->depthBufferActive) {
            new_fbo.BindDepthTexture();
            glClear(GL_DEPTH_BUFFER_BIT);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                _framebuffer->depthBuffer.data());
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        float right = (width + static_cast<float>(width)) / 2.0f;
        float left = (width - static_cast<float>(width)) / 2.0f;
        float bottom = (height + static_cast<float>(height)) / 2.0f;
        float up = (height - static_cast<float>(height)) / 2.0f;
        glm::vec3 pos_bottom_left = {left, bottom, 0.0f};
        glm::vec3 pos_upper_left = {left, up, 0.0f};
        glm::vec3 pos_upper_right = {right, up, 0.0f};
        glm::vec3 pos_bottom_right = {right, bottom, 0.0f};
        _utils.Push2DColorTexture(
            new_fbo.GetColourTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right,true);
        if (_framebuffer->depthBufferActive) {
            _utils.Push2DDepthTexture(
                new_fbo.GetDepthTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
        }

        new_fbo.Disable();

        // draw into lhs fbo
        if ((lhs_fbo->GetWidth() != width) || (lhs_fbo->GetHeight() != height)) {
            lhs_fbo->Release();
            lhs_fbo->Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
        }
        if (lhs_fbo->IsValid() && !lhs_fbo->IsEnabled()) {
            lhs_fbo->Enable();
        }

        glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);

        _utils.DrawTextures(ortho, glm::vec2(width, height));

        if (lhs_fbo->IsValid()) {
            lhs_fbo->Disable();
        }
    } else {
        return false;
    }
    return true;
}

} // namespace megamol::core::view
