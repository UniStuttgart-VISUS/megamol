#include "stdafx.h"
#include "TextureCombine.h"

#include <array>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::TextureCombine::TextureCombine()
    : core::Module()
    , m_output_texture(nullptr)
    , m_output_texture_hash(0)
    , m_mode("Mode", "Sets texture combination mode, e.g. add, multiply...")
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_0_slot(
          "InputTexture0", "Connects the primary input texture that is also used the set the output texture size")
    , m_input_tex_1_slot("InputTexture1", "Connects the secondary input texture") {
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Add");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Multiply");
    this->MakeSlotAvailable(&this->m_mode);

    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetData", &TextureCombine::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &TextureCombine::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_0_slot);

    this->m_input_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_1_slot);
}

megamol::compositing::TextureCombine::~TextureCombine() { this->Release(); }

bool megamol::compositing::TextureCombine::create() {

    try {
        // create shader program
        m_add_prgm = std::make_unique<GLSLComputeShader>();
        m_mult_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_add_src;
        vislib::graphics::gl::ShaderSource compute_mult_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::textureAdd", compute_add_src))
            return false;
        if (!m_add_prgm->Compile(compute_add_src.Code(), compute_add_src.Count())) return false;
        if (!m_add_prgm->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::textureMultiply", compute_mult_src))
            return false;
        if (!m_mult_prgm->Compile(compute_mult_src.Code(), compute_mult_src.Count())) return false;
        if (!m_mult_prgm->Link()) return false;

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("textureCombine_output", tx_layout, nullptr);

    return true;
}

void megamol::compositing::TextureCombine::release() {}

bool megamol::compositing::TextureCombine::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto rhs_tc0 = m_input_tex_0_slot.CallAs<CallTexture2D>();
    auto rhs_tc1 = m_input_tex_1_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    if (rhs_tc0 == NULL) return false;
    if (rhs_tc1 == NULL) return false;

    if (!(*rhs_tc0)(0)) return false;
    if (!(*rhs_tc1)(0)) return false;

    // TODO check/update data hash

    if (lhs_tc->getData() == nullptr) {
        lhs_tc->setData(m_output_texture);
    }

    // set output texture size to primary input texture
    auto src0_tx2D = rhs_tc0->getData();
    auto src1_tx2D = rhs_tc1->getData();
    std::array<float, 2> texture_res = {
        static_cast<float>(src0_tx2D->getWidth()), static_cast<float>(src0_tx2D->getHeight())};

    if (m_output_texture->getWidth() != std::get<0>(texture_res) ||
        m_output_texture->getHeight() != std::get<1>(texture_res)) {
        glowl::TextureLayout tx_layout(
            GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_RGBA, GL_HALF_FLOAT, 1);
        m_output_texture->reload(tx_layout, nullptr);
    }

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        m_add_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        src0_tx2D->bindTexture();
        glUniform1i(m_add_prgm->ParameterLocation("src0_tx2D"), 0);
        glActiveTexture(GL_TEXTURE1);
        src1_tx2D->bindTexture();
        glUniform1i(m_add_prgm->ParameterLocation("src1_tx2D"), 1);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);

        m_add_prgm->Dispatch(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
            static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

        m_add_prgm->Disable();
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        m_mult_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        src0_tx2D->bindTexture();
        glUniform1i(m_add_prgm->ParameterLocation("src0_tx2D"), 0);
        glActiveTexture(GL_TEXTURE1);
        src1_tx2D->bindTexture();
        glUniform1i(m_add_prgm->ParameterLocation("src1_tx2D"), 1);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);

        m_mult_prgm->Dispatch(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
            static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

        m_mult_prgm->Disable();
    }

    return true;
}

bool megamol::compositing::TextureCombine::getMetaDataCallback(core::Call& caller) {

    // TODO output hash?

    return true;
}
