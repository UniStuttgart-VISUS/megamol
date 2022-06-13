#include "DrawScalarTexture.h"

#include "mmcore/CoreInstance.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"

#include "compositing_gl/CompositingCalls.h"


megamol::benchmark_gl::DrawScalarTexture::DrawScalarTexture() : tex_in_slot_("texIn", ""), tf_slot_("tfIn", "") {
    tex_in_slot_.SetCompatibleCall<compositing::CallTexture2DDescription>();
    MakeSlotAvailable(&tex_in_slot_);

    tf_slot_.SetCompatibleCall<core_gl::view::CallGetTransferFunctionGLDescription>();
    MakeSlotAvailable(&tf_slot_);
}


megamol::benchmark_gl::DrawScalarTexture::~DrawScalarTexture() {
    this->Release();
}


bool megamol::benchmark_gl::DrawScalarTexture::create() {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        shader_ = core::utility::make_glowl_shader("draw_scalar_texture", shader_options,
            "benchmark_gl/draw_scalar.vert.glsl", "benchmark_gl/draw_scalar.frag.glsl");
    } catch (const std::exception& e) {
        Log::DefaultLog.WriteError(("DrawScalarTexture: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


void megamol::benchmark_gl::DrawScalarTexture::release() {}


bool megamol::benchmark_gl::DrawScalarTexture::Render(core_gl::view::CallRender2DGL& call) {
    auto cr = &call;
    if (cr == nullptr)
        return false;

    auto ct = tex_in_slot_.CallAs<compositing::CallTexture2D>();
    if (ct == NULL)
        return false;
    (*ct)(0);

    auto width = call.GetFramebuffer()->getWidth();
    auto height = call.GetFramebuffer()->getHeight();

    // get input texture from call
    auto input_texture = ct->getData();
    if (input_texture == nullptr)
        return false;

    if (!updateTransferFunction())
        return false;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    if (shader_ != nullptr) {
        shader_->use();

        glActiveTexture(GL_TEXTURE0);
        input_texture->bindTexture();
        glUniform1i(shader_->getUniformLocation("input_tx2D"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tf_texture);
        glUniform1i(shader_->getUniformLocation("tf_tx1D"), 1);

        glUniform2i(shader_->getUniformLocation("viewport_resolution"), width, height);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(0);
    }

    glDisable(GL_BLEND);

    return true;
}


bool megamol::benchmark_gl::DrawScalarTexture::GetExtents(core_gl::view::CallRender2DGL& call) {
    return true;
}


bool megamol::benchmark_gl::DrawScalarTexture::OnKey(
    core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    return false;
}


bool megamol::benchmark_gl::DrawScalarTexture::OnChar(unsigned int codePoint) {
    return false;
}


bool megamol::benchmark_gl::DrawScalarTexture::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    return false;
}


bool megamol::benchmark_gl::DrawScalarTexture::OnMouseMove(double x, double y) {
    return false;
}


bool megamol::benchmark_gl::DrawScalarTexture::OnMouseScroll(double dx, double dy) {
    return false;
}


bool megamol::benchmark_gl::DrawScalarTexture::updateTransferFunction() {
    core_gl::view::CallGetTransferFunctionGL* ct =
        this->tf_slot_.CallAs<core_gl::view::CallGetTransferFunctionGL>();
    /*if (valRangeNeedsUpdate) {
        ct->SetRange(valRange);
        valRangeNeedsUpdate = false;
    }*/
    if (ct != nullptr && ((*ct)())) {
        tf_texture = ct->OpenGLTexture();
        //valRange = ct->Range();
    }

    return true;
}
