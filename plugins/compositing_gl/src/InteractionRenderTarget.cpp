#include "InteractionRenderTarget.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::InteractionRenderTarget::InteractionRenderTarget() 
    : SimpleRenderTarget(), m_objId_render_target("ObjectId","Access the object id render target texture") 
{
    this->m_objId_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetData", &InteractionRenderTarget::getObjectIdRenderTarget);
    this->m_objId_render_target.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &InteractionRenderTarget::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_objId_render_target);
}

bool megamol::compositing::InteractionRenderTarget::create() {

    SimpleRenderTarget::create();

    m_GBuffer->createColorAttachment(GL_R32I, GL_RED, GL_INT); // object ids
    m_GBuffer->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // additional interaction information, e.g. clicked screen value

    return true; 
}

bool megamol::compositing::InteractionRenderTarget::Render(core::view::CallRender3D_2& call) {

    SimpleRenderTarget::Render(call);

    GLint in[1] = {-1};
    glClearBufferiv(GL_COLOR, 3, in);

    return true;
}

bool megamol::compositing::InteractionRenderTarget::getObjectIdRenderTarget(core::Call& caller) { 
    auto ct = dynamic_cast<CallTexture2D*>(&caller);

    if (ct == NULL) return false;

    ct->setData(m_GBuffer->getColorAttachment(3), this->m_version);

    return true;
}

bool megamol::compositing::InteractionRenderTarget::getMetaDataCallback(core::Call& caller) { return true; }
