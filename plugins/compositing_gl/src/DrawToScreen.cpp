#include "stdafx.h"

#include "DrawToScreen.h"

#include "mmcore/CoreInstance.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore_gl/flags/FlagCallsGL.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"

megamol::compositing::DrawToScreen::DrawToScreen()
        : core_gl::view::Renderer3DModuleGL()
        , m_dummy_color_tx(nullptr)
        , m_dummy_depth_tx(nullptr)
        , m_drawToScreen_prgm(nullptr)
        , m_input_texture_call("InputTexture", "Access texture that is drawn to output screen")
        , m_input_depth_texture_call("DepthTexture", "Access optional depth texture to write depth values to screen")
        , m_input_flags_call("readFlagStorage", "Flag storage read input") {
    this->m_input_texture_call.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_texture_call);

    this->m_input_depth_texture_call.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_depth_texture_call);

    m_input_flags_call.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    MakeSlotAvailable(&m_input_flags_call);
}

megamol::compositing::DrawToScreen::~DrawToScreen() {
    this->Release();
}

bool megamol::compositing::DrawToScreen::create() {

    // create shader program
    vislib_gl::graphics::gl::ShaderSource vert_shader_src;
    vislib_gl::graphics::gl::ShaderSource frag_shader_src;

    vislib::StringA shader_base_name("comp_drawToScreen");
    vislib::StringA vertShaderName = shader_base_name + "::vertex";
    vislib::StringA fragShaderName = shader_base_name + "::fragment";

    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
    ssf->MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    ssf->MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);

    try {
        m_drawToScreen_prgm = std::make_unique<GLSLShader>();
        m_drawToScreen_prgm->Create(
            vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile %s (@%s):\n%s\n", shader_base_name.PeekBuffer(),
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        // return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile %s:\n%s\n", shader_base_name.PeekBuffer(), e.GetMsgA());
        // return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile %s: Unknown exception\n", shader_base_name.PeekBuffer());
        // return false;
    }

    auto err = glGetError();

    glowl::TextureLayout depth_tx_layout(GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT, 1);
    std::array<float, 1> dummy_depth_data = {-1.0f};
    m_dummy_depth_tx =
        std::make_shared<glowl::Texture2D>("DrawToScreen_dummyDepth", depth_tx_layout, dummy_depth_data.data());

    glowl::TextureLayout color_tx_layout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1);
    std::array<uint8_t, 4> dummy_color_data = {0, 0, 0, 0};
    m_dummy_depth_tx =
        std::make_shared<glowl::Texture2D>("DrawToScreen_dummyColor", color_tx_layout, dummy_color_data.data());

    return true;
}

void megamol::compositing::DrawToScreen::release() {
    m_drawToScreen_prgm.reset();
}

bool megamol::compositing::DrawToScreen::GetExtents(core_gl::view::CallRender3DGL& call) {
    return true;
}

bool megamol::compositing::DrawToScreen::Render(core_gl::view::CallRender3DGL& call) {
    // get lhs render call
    megamol::core_gl::view::CallRender3DGL* cr = &call;
    if (cr == NULL)
        return false;

    // get rhs texture call
    std::shared_ptr<glowl::Texture2D> color_texture = m_dummy_color_tx;
    CallTexture2D* ct = this->m_input_texture_call.CallAs<CallTexture2D>();
    if (ct != NULL) {
        (*ct)(0);
        color_texture = ct->getData();
    }

    // get rhs depth texture call
    std::shared_ptr<glowl::Texture2D> depth_texture = m_dummy_depth_tx;
    CallTexture2D* cdt = this->m_input_depth_texture_call.CallAs<CallTexture2D>();
    if (cdt != NULL) {
        (*cdt)(0);
        depth_texture = cdt->getData();
    }

    if (color_texture == nullptr || depth_texture == nullptr) {
        return false;
    }

    auto width = call.GetFramebuffer()->getWidth();
    auto height = call.GetFramebuffer()->getHeight();

    auto readFlagsCall = m_input_flags_call.CallAs<core_gl::FlagCallRead_GL>();
    if (readFlagsCall != nullptr) {
        (*readFlagsCall)(core_gl::FlagCallRead_GL::CallGetData);

        if (m_last_tex_size != glm::ivec2(color_texture->getWidth(), color_texture->getHeight()) ||
            readFlagsCall->hasUpdate()) {
            readFlagsCall->getData()->validateFlagCount(color_texture->getWidth() * color_texture->getHeight());
            m_last_tex_size = glm::ivec2(color_texture->getWidth(), color_texture->getHeight());
        }
        readFlagsCall->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 5);
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    if (m_drawToScreen_prgm != nullptr) {
        m_drawToScreen_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        color_texture->bindTexture();
        glUniform1i(m_drawToScreen_prgm->ParameterLocation("input_tx2D"), 0);

        glActiveTexture(GL_TEXTURE1);
        depth_texture->bindTexture();
        glUniform1i(m_drawToScreen_prgm->ParameterLocation("depth_tx2D"), 1);

        glUniform1ui(m_drawToScreen_prgm->ParameterLocation("flags_available"), readFlagsCall != nullptr ? 1 : 0);
        glUniform1ui(m_drawToScreen_prgm->ParameterLocation("frame_id"), this->GetCoreInstance()->GetFrameID());
        glUniform2i(m_drawToScreen_prgm->ParameterLocation("viewport_resolution"), width, height);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        m_drawToScreen_prgm->Disable();
    }

    glDisable(GL_BLEND);

    return true;
}

void megamol::compositing::DrawToScreen::PreRender(core_gl::view::CallRender3DGL& call) {}
