#include "stdafx.h"

#include "DrawToScreen.h"

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::DrawToScreen::DrawToScreen() 
    : Renderer3DModuleGL()
    , m_drawToScreen_prgm(nullptr)
    , m_input_texture_call("InputTexture","Access texture that is drawn to output screen")
{
    this->m_input_texture_call.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_texture_call);
}

megamol::compositing::DrawToScreen::~DrawToScreen() { this->Release(); }

bool megamol::compositing::DrawToScreen::create() {

    // create shader program
    vislib::graphics::gl::ShaderSource vert_shader_src;
    vislib::graphics::gl::ShaderSource frag_shader_src;

    vislib::StringA shader_base_name("comp_drawToScreen");
    vislib::StringA vertShaderName = shader_base_name + "::vertex";
    vislib::StringA fragShaderName = shader_base_name + "::fragment";

    this->instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    this->instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);

    try {
        m_drawToScreen_prgm = std::make_unique<GLSLShader>();
        m_drawToScreen_prgm->Create(
            vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            shader_base_name.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        // return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile %s:\n%s\n", shader_base_name.PeekBuffer(), e.GetMsgA());
        // return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", shader_base_name.PeekBuffer());
        // return false;
    }

    //m_drawToScreen_prgm

    return true; 
}

void megamol::compositing::DrawToScreen::release() {
    m_drawToScreen_prgm.reset(); 
}

bool megamol::compositing::DrawToScreen::GetExtents(core::view::CallRender3DGL& call) { 
    return true; 
}

bool megamol::compositing::DrawToScreen::Render(core::view::CallRender3DGL& call) { 
    // get lhs render call
    megamol::core::view::CallRender3DGL* cr = &call;
    if (cr == NULL) return false;

    // Restore framebuffer that was bound on the way in
    glBindFramebuffer(GL_FRAMEBUFFER, m_screenRestoreFBO);

    // get rhs texture call
    CallTexture2D* ct = this->m_input_texture_call.CallAs<CallTexture2D>();
    if (ct == NULL) return false;
    (*ct)(0);

    // obtain camera information
    //  core::view::Camera_2 cam(cr->GetCamera());
    //  cam_type::snapshot_type snapshot;
    //  cam_type::matrix_type view_tmp, proj_tmp;
    //  cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
    //  glm::mat4 view_mx = view_tmp;
    //  glm::mat4 proj_mx = proj_tmp;

    // get input texture from call
    auto input_texture = ct->getData();
    if (input_texture == nullptr) return false;

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    if (m_drawToScreen_prgm != nullptr) {
        m_drawToScreen_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        input_texture->bindTexture();
        glUniform1i(m_drawToScreen_prgm->ParameterLocation("input_tx2D"), 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        m_drawToScreen_prgm->Disable();
    }

    glDisable(GL_BLEND);

    return true; 
}

void megamol::compositing::DrawToScreen::PreRender(core::view::CallRender3DGL& call) {

    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &m_screenRestoreFBO);

}
