#include "stdafx.h"

#include "SimpleRenderTarget.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::SimpleRenderTarget::SimpleRenderTarget() 
    : Renderer3DModule_2()
    , m_GBuffer(nullptr) 
    , m_color_render_target("Color", "Acces the color render target texture")
    , m_normal_render_target("Normals", "Acces the normals render target texture")
    , m_depth_render_target("Depth", "Access the depth render target texture") 
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
}

megamol::compositing::SimpleRenderTarget::~SimpleRenderTarget() { 
    m_GBuffer.reset(); 
    this->Release();
}

bool megamol::compositing::SimpleRenderTarget::create() { 
    return true; 
}

void megamol::compositing::SimpleRenderTarget::release() {
}

bool megamol::compositing::SimpleRenderTarget::GetExtents(core::view::CallRender3D_2& call) { 
    return true; 
}

bool megamol::compositing::SimpleRenderTarget::Render(core::view::CallRender3D_2& call) { 
    return false; 
}

void megamol::compositing::SimpleRenderTarget::PreRender(core::view::CallRender3D_2& call)
{
    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    if (m_GBuffer == nullptr) {
        m_GBuffer = std::make_unique<glowl::FramebufferObject>(viewport[2], viewport[3], true);
        m_GBuffer->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT); // surface albedo
        m_GBuffer->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT); // normals
        m_GBuffer->createColorAttachment(GL_R32F, GL_RED, GL_FLOAT);        // clip space depth
    } else if (m_GBuffer->getWidth() != viewport[2] || m_GBuffer->getHeight() != viewport[3]) {
        m_GBuffer->resize(viewport[2], viewport[3]);
    }

    m_GBuffer->bind();

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

bool megamol::compositing::SimpleRenderTarget::getMetaDataCallback(core::Call& caller) { return false; }
