#include "stdafx.h"

#include "SimpleRenderTarget.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::SimpleRenderTarget::SimpleRenderTarget() 
    : Renderer3DModule_2()
    , m_GBuffer(nullptr)
    , m_color_render_target("Color", "Acces the color render target texture")
    , m_normal_render_target("Normals", "Acces the normals render target texture")
    , m_depth_render_target("Depth", "Access the depth render target texture")
    , m_camera("Camera", "Access the latest camera snapshot")
    , m_framebuffer_slot("Framebuffer", "Publish the framebuffer used by this render target")
{
    this->m_color_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetData", &SimpleRenderTarget::getColorRenderTarget);
    this->m_color_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_color_render_target);

    this->m_normal_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetData", &SimpleRenderTarget::getNormalsRenderTarget);
    this->m_normal_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_normal_render_target);

    this->m_depth_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetData", &SimpleRenderTarget::getDepthRenderTarget);
    this->m_depth_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_depth_render_target);

    this->m_camera.SetCallback(
        CallCamera::ClassName(), "GetData", &SimpleRenderTarget::getCameraSnapshot);
    this->m_camera.SetCallback(
        CallCamera::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_camera);

    this->m_framebuffer_slot.SetCompatibleCall<CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_framebuffer_slot);
}

megamol::compositing::SimpleRenderTarget::~SimpleRenderTarget() { 
    m_GBuffer.reset(); 
    this->Release();
}

bool megamol::compositing::SimpleRenderTarget::create() { 

    m_GBuffer = std::make_shared<glowl::FramebufferObject>(1, 1, true);
    m_GBuffer->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // surface albedo
    m_GBuffer->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT); // normals
    m_GBuffer->createColorAttachment(GL_R32F, GL_RED, GL_FLOAT);        // clip space depth

    return true; 
}

void megamol::compositing::SimpleRenderTarget::release() {
}

bool megamol::compositing::SimpleRenderTarget::GetExtents(core::view::CallRender3D_2& call) { 
    return true; 
}

bool megamol::compositing::SimpleRenderTarget::Render(core::view::CallRender3D_2& call) {
    glClearColor(this->clear_color[0], this->clear_color[1], this->clear_color[2], this->clear_color[3]);

    glBindFramebuffer(GL_FRAMEBUFFER_EXT, this->old_fb);
    glDrawBuffer(this->old_db);
    glReadBuffer(this->old_rb);

    return false; 
}

void megamol::compositing::SimpleRenderTarget::PreRender(core::view::CallRender3D_2& call)
{
    m_last_used_camera = call.GetCamera();

    auto rhs_fc = m_framebuffer_slot.CallAs<CallFramebufferGL>();
    if (rhs_fc != NULL)
    {
        rhs_fc->setData(m_GBuffer);
        (*rhs_fc)(0);
    }

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &this->old_fb);
    glGetIntegerv(GL_DRAW_BUFFER, &this->old_db);
    glGetIntegerv(GL_READ_BUFFER, &this->old_rb);

    if (m_GBuffer->getWidth() != viewport[2] || m_GBuffer->getHeight() != viewport[3]) {
        m_GBuffer->resize(viewport[2], viewport[3]);
    }

    m_GBuffer->bind();

    glGetFloatv(GL_COLOR_CLEAR_VALUE, this->clear_color.data());
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

bool megamol::compositing::SimpleRenderTarget::getColorRenderTarget(core::Call& caller) {
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL) return false;

    ct->setData(m_GBuffer->getColorAttachment(0));

    return true;
}

bool megamol::compositing::SimpleRenderTarget::getNormalsRenderTarget(core::Call& caller) {
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL) return false;

    ct->setData(m_GBuffer->getColorAttachment(1));

    return true;
}

bool megamol::compositing::SimpleRenderTarget::getDepthRenderTarget(core::Call& caller) { 
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL) return false;

    ct->setData(m_GBuffer->getColorAttachment(2));

    return true;
}

bool megamol::compositing::SimpleRenderTarget::getCameraSnapshot(core::Call& caller) { 
    auto ct = dynamic_cast<CallCamera*>(&caller);

    if (ct == NULL) return false;

    ct->setData(m_last_used_camera);

    return true; 
}

bool megamol::compositing::SimpleRenderTarget::getMetaDataCallback(core::Call& caller) { return false; }
