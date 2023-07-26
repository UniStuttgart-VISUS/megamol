
#include "SimpleRenderTarget.h"

#include "compositing_gl/CompositingCalls.h"

megamol::compositing_gl::SimpleRenderTarget::SimpleRenderTarget()
        : core::view::RendererModule<mmstd_gl::CallRender3DGL, ModuleGL>()
        , m_version(0)
        , m_GBuffer(nullptr)
        , m_last_used_camera(glm::mat4(1.0f), glm::mat4(1.0f))
        , m_color_render_target("Color", "Access the color render target texture")
        , m_normal_render_target("Normals", "Access the normals render target texture")
        , m_depth_buffer("Depth", "Access the depth render target texture")
        , m_camera("Camera", "Access the latest camera snapshot")
        , m_framebuffer_slot("Framebuffer", "Access the framebuffer used by this render target") {
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

    this->m_depth_buffer.SetCallback(CallTexture2D::ClassName(), "GetData", &SimpleRenderTarget::getDepthBuffer);
    this->m_depth_buffer.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_depth_buffer);

    this->m_camera.SetCallback(CallCamera::ClassName(), "GetData", &SimpleRenderTarget::getCameraSnapshot);
    this->m_camera.SetCallback(CallCamera::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_camera);

    this->m_framebuffer_slot.SetCallback(
        CallFramebufferGL::ClassName(), "GetData", &SimpleRenderTarget::getFramebufferObject);
    this->m_framebuffer_slot.SetCallback(
        CallFramebufferGL::ClassName(), "GetMetaData", &SimpleRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_framebuffer_slot);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

megamol::compositing_gl::SimpleRenderTarget::~SimpleRenderTarget() {
    m_GBuffer.reset();
    this->Release();
}

bool megamol::compositing_gl::SimpleRenderTarget::create() {

    m_GBuffer = std::make_shared<glowl::FramebufferObject>(1, 1);
    m_GBuffer->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // surface albedo
    m_GBuffer->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // normals
    //m_GBuffer->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT);   // normals

    return true;
}

void megamol::compositing_gl::SimpleRenderTarget::release() {}

bool megamol::compositing_gl::SimpleRenderTarget::GetExtents(mmstd_gl::CallRender3DGL& call) {
    mmstd_gl::CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
    if (chainedCall != nullptr) {
        *chainedCall = call;
        bool retVal = (*chainedCall)(core::view::AbstractCallRender::FnGetExtents);
        call = *chainedCall;
    }
    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::Render(mmstd_gl::CallRender3DGL& call) {
    ++m_version;

    m_last_used_camera = call.GetCamera();

    auto call_fbo = call.GetFramebuffer();

    if (m_GBuffer->getWidth() != call_fbo->getWidth() || m_GBuffer->getHeight() != call_fbo->getHeight()) {
        m_GBuffer->resize(call_fbo->getWidth(), call_fbo->getHeight());
    }

    // this framebuffer will use 0 clear color because it uses alpha transparency during
    // compositing and final presentation to screen anyway
    m_GBuffer->bind();
    glClearColor(0, 0, 0, 0);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    mmstd_gl::CallRender3DGL* chained_call = this->chainRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
    if (chained_call != nullptr) {
        *chained_call = call;

        chained_call->SetFramebuffer(m_GBuffer);

        if (!((*chained_call)(core::view::AbstractCallRender::FnRender))) {
            return false;
        }
    }

    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::getColorRenderTarget(core::Call& caller) {
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL)
        return false;

    ct->setData(m_GBuffer->getColorAttachment(0), m_version);

    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::getNormalsRenderTarget(core::Call& caller) {
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL)
        return false;

    ct->setData(m_GBuffer->getColorAttachment(1), m_version);

    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::getDepthBuffer(core::Call& caller) {
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL)
        return false;

    ct->setData(m_GBuffer->getDepthStencil(), m_version);

    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::getCameraSnapshot(core::Call& caller) {
    auto cc = dynamic_cast<CallCamera*>(&caller);

    if (cc == NULL)
        return false;

    cc->setData(m_last_used_camera, m_version);

    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::getFramebufferObject(core::Call& caller) {
    auto cf = dynamic_cast<CallFramebufferGL*>(&caller);

    if (cf == NULL)
        return false;

    cf->setData(m_GBuffer, m_version);

    return true;
}

bool megamol::compositing_gl::SimpleRenderTarget::getMetaDataCallback(core::Call& caller) {
    return true;
}
