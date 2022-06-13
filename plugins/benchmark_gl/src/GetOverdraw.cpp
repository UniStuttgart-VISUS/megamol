#include "GetOverdraw.h"

#include "mmcore_gl/view/CallRender3DGL.h"
#include "compositing_gl/CompositingCalls.h"


megamol::benchmark_gl::GetOverdraw::GetOverdraw(void)
        : get_tex_slot_("getTex", "")
        , render_out_slot_("renderOut", "")
        , render_in_slot_("renderIn", "") {
    get_tex_slot_.SetCallback(
        compositing::CallTexture2D::ClassName(), compositing::CallTexture2D::FunctionName(0), &GetOverdraw::get_tex);
    get_tex_slot_.SetCallback(compositing::CallTexture2D::ClassName(), compositing::CallTexture2D::FunctionName(1),
        &GetOverdraw::get_tex_meta);
    MakeSlotAvailable(&get_tex_slot_);

    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(0), &GetOverdraw::passthrough);
    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(1), &GetOverdraw::passthrough);
    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(2), &GetOverdraw::passthrough);
    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(3), &GetOverdraw::passthrough);
    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(4), &GetOverdraw::passthrough);
    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(core_gl::view::CallRender3DGL::FnRender), &GetOverdraw::Render);
    render_out_slot_.SetCallback(core_gl::view::CallRender3DGL::ClassName(),
        core_gl::view::CallRender3DGL::FunctionName(core_gl::view::CallRender3DGL::FnGetExtents), &GetOverdraw::GetExtents);
    MakeSlotAvailable(&render_out_slot_);

    render_in_slot_.SetCompatibleCall<core_gl::view::CallRender3DGLDescription>();
    MakeSlotAvailable(&render_in_slot_);
}


megamol::benchmark_gl::GetOverdraw::~GetOverdraw(void) {
    this->Release();
}


bool megamol::benchmark_gl::GetOverdraw::create() {
    return true;
}


void megamol::benchmark_gl::GetOverdraw::release() {}


bool megamol::benchmark_gl::GetOverdraw::GetExtents(core::Call& c) {
    auto call = dynamic_cast<core_gl::view::CallRender3DGL*>(&c);
    if (call == nullptr)
        return false;

    auto chainedCall = render_in_slot_.CallAs<core_gl::view::CallRender3DGL>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = *call;

        // chain through the get extents call
        (*chainedCall)(core::view::AbstractCallRender::FnGetExtents);
    }

    // TODO extents magic


    // get our own extents
    // this->GetExtents(call);

    if (chainedCall != nullptr) {
        auto mybb = call->AccessBoundingBoxes().BoundingBox();
        auto otherbb = chainedCall->AccessBoundingBoxes().BoundingBox();
        auto mycb = call->AccessBoundingBoxes().ClipBox();
        auto othercb = chainedCall->AccessBoundingBoxes().ClipBox();

        if (call->AccessBoundingBoxes().IsBoundingBoxValid() &&
            chainedCall->AccessBoundingBoxes().IsBoundingBoxValid()) {
            mybb.Union(otherbb);
        } else if (chainedCall->AccessBoundingBoxes().IsBoundingBoxValid()) {
            mybb = otherbb; // just override for the call
        }                   // we ignore the other two cases as they both lead to usage of the already set mybb

        if (call->AccessBoundingBoxes().IsClipBoxValid() && chainedCall->AccessBoundingBoxes().IsClipBoxValid()) {
            mycb.Union(othercb);
        } else if (chainedCall->AccessBoundingBoxes().IsClipBoxValid()) {
            mycb = othercb; // just override for the call
        }                   // we ignore the other two cases as they both lead to usage of the already set mycb

        call->AccessBoundingBoxes().SetBoundingBox(mybb);
        call->AccessBoundingBoxes().SetClipBox(mycb);

        // TODO machs richtig
        call->SetTimeFramesCount(chainedCall->TimeFramesCount());
    }

    return true;
}


void go_blit_fbo(std::shared_ptr<glowl::FramebufferObject>& org, std::shared_ptr<glowl::FramebufferObject>& dest) {
    org->bindToRead(0);
    dest->bindToDraw();
    glBlitFramebuffer(0, 0, org->getWidth(), org->getHeight(), 0, 0, dest->getWidth(), dest->getHeight(),
        GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}


void copy_stencil(std::shared_ptr<glowl::FramebufferObject>& org, std::shared_ptr<glowl::Texture2D>& dest) {
    org->bind();
    /*dest->bindTexture();
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_STENCIL, 0, 0, dest->getWidth(), dest->getHeight(), 0);
    glBindTexture(GL_TEXTURE_2D, 0);*/
    std::vector<uint8_t> data(org->getWidth() * org->getHeight());
    glReadPixels(0, 0, org->getWidth(), org->getHeight(), GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, data.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


bool megamol::benchmark_gl::GetOverdraw::Render(core::Call& c) {
    auto call = dynamic_cast<core_gl::view::CallRender3DGL*>(&c);
    if (call == nullptr)
        return false;

    auto chain_call = render_in_slot_.CallAs<core_gl::view::CallRender3DGL>();
    if (chain_call == nullptr)
        return false;

    *chain_call = *call;

    auto in_fbo = call->GetFramebuffer();
    if (!rt_ || rt_->getWidth() != in_fbo->getWidth() || rt_->getHeight() != in_fbo->getHeight()) {
        if (rt_) {
            rt_->resize(in_fbo->getWidth(), in_fbo->getHeight());
        } else {
            rt_ = std::make_shared<glowl::FramebufferObject>(
                in_fbo->getWidth(), in_fbo->getHeight(), glowl::FramebufferObject::DEPTH24_STENCIL8);
            rt_->createColorAttachment(GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
        }
    }

    if (!stencil_buffer_copy_ || stencil_buffer_copy_->getWidth() != in_fbo->getWidth() ||
        stencil_buffer_copy_->getHeight() != in_fbo->getHeight()) {
        auto layout = glowl::TextureLayout(GL_STENCIL_INDEX, in_fbo->getWidth(), in_fbo->getHeight(), 1,
            GL_STENCIL_INDEX,
            GL_UNSIGNED_BYTE, 1,
            {{GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}, {GL_TEXTURE_WRAP_S, GL_REPEAT},
                {GL_TEXTURE_WRAP_T, GL_REPEAT}},
            {});
        stencil_buffer_copy_ = std::make_shared<glowl::Texture2D>("stencil_copy", layout, nullptr);
    }

    if (!pbo_ || pbo_->getByteSize() != in_fbo->getWidth()*in_fbo->getHeight()) {
        pbo_ = std::make_shared<glowl::BufferObject>(
            GL_PIXEL_PACK_BUFFER, nullptr, in_fbo->getWidth() * in_fbo->getHeight(), GL_STREAM_READ);
    }

    chain_call->SetFramebuffer(rt_);

    rt_->bind();
    glViewport(0, 0, rt_->getWidth(), rt_->getHeight());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilMask(0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
    (*chain_call)(core_gl::view::CallRender3DGL::FnRender);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glDisable(GL_STENCIL_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // TODO Blit Framebuffer
    go_blit_fbo(rt_, in_fbo);
    chain_call->SetFramebuffer(in_fbo);

    *call = *chain_call;

    //copy_stencil(rt_, stencil_buffer_copy_);
    rt_->bind();
    pbo_->bind();
    glReadPixels(0, 0, rt_->getWidth(), rt_->getHeight(), GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_->getName());
    stencil_buffer_copy_->bindTexture();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, stencil_buffer_copy_->getWidth(), stencil_buffer_copy_->getHeight(),
        GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    ++version_;

    /*glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    stencil_buffer_copy_->bindTexture();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);*/

    return true;
}


bool megamol::benchmark_gl::GetOverdraw::get_tex(core::Call& c) {
    auto ct = dynamic_cast<compositing::CallTexture2D*>(&c);

    if (ct == nullptr)
        return false;

    ct->setData(stencil_buffer_copy_, version_);

    return true;
}


bool megamol::benchmark_gl::GetOverdraw::get_tex_meta(core::Call& c) {
    return true;
}


bool megamol::benchmark_gl::GetOverdraw::passthrough(core::Call& c) {
    return false;
}
