
#include "DrawToScreen.h"

#include "FrameStatistics.h"
#include "compositing_gl/CompositingCalls.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/flags/FlagCallsGL.h"

using megamol::core::utility::log::Log;

megamol::compositing_gl::DrawToScreen::DrawToScreen()
        : mmstd_gl::Renderer3DModuleGL()
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

    m_input_flags_call.SetCompatibleCall<mmstd_gl::FlagCallRead_GLDescription>();
    MakeSlotAvailable(&m_input_flags_call);
}

megamol::compositing_gl::DrawToScreen::~DrawToScreen() {
    this->Release();
}

bool megamol::compositing_gl::DrawToScreen::create() {

    // create shader program
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        m_drawToScreen_prgm = core::utility::make_glowl_shader("comp_drawToScreen", shader_options,
            "compositing_gl/drawToScreen.vert.glsl", "compositing_gl/drawToScreen.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("DrawToScreen: " + std::string(e.what())).c_str());
        return false;
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

void megamol::compositing_gl::DrawToScreen::release() {
    m_drawToScreen_prgm.reset();
}

bool megamol::compositing_gl::DrawToScreen::GetExtents(mmstd_gl::CallRender3DGL& call) {
    return true;
}

bool megamol::compositing_gl::DrawToScreen::Render(mmstd_gl::CallRender3DGL& call) {
    // get lhs render call
    mmstd_gl::CallRender3DGL* cr = &call;
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

    auto readFlagsCall = m_input_flags_call.CallAs<mmstd_gl::FlagCallRead_GL>();
    if (readFlagsCall != nullptr) {
        (*readFlagsCall)(mmstd_gl::FlagCallRead_GL::CallGetData);

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
        m_drawToScreen_prgm->use();

        glActiveTexture(GL_TEXTURE0);
        color_texture->bindTexture();
        glUniform1i(m_drawToScreen_prgm->getUniformLocation("input_tx2D"), 0);

        glActiveTexture(GL_TEXTURE1);
        depth_texture->bindTexture();
        glUniform1i(m_drawToScreen_prgm->getUniformLocation("depth_tx2D"), 1);

        glUniform1ui(m_drawToScreen_prgm->getUniformLocation("flags_available"), readFlagsCall != nullptr ? 1 : 0);
        glUniform1ui(m_drawToScreen_prgm->getUniformLocation("frame_id"),
            frontend_resources.get<frontend_resources::FrameStatistics>().rendered_frames_count);
        glUniform2i(m_drawToScreen_prgm->getUniformLocation("viewport_resolution"), width, height);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(0);
    }

    glDisable(GL_BLEND);

    return true;
}

void megamol::compositing_gl::DrawToScreen::PreRender(mmstd_gl::CallRender3DGL& call) {}
