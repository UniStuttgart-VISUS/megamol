#include "TextureCombine.h"

#include <array>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"

#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"

megamol::compositing::TextureCombine::TextureCombine()
        : core::Module()
        , m_version(0)
        , m_output_texture(nullptr)
        , m_mode("Mode", "Sets texture combination mode, e.g. add, multiply...")
        , m_weight_0("Weight0", "Weight for input texture 0 in additive mode")
        , m_weight_1("Weight1", "Weight for input texture 1 in additive mode")
        , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
        , m_input_tex_0_slot(
              "InputTexture0", "Connects the primary input texture that is also used the set the output texture size")
        , m_input_tex_1_slot("InputTexture1", "Connects the secondary input texture") {
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Add");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Multiply");
    this->m_mode.SetUpdateCallback(&TextureCombine::modeCallback);
    this->MakeSlotAvailable(&this->m_mode);

    this->m_weight_0 << new megamol::core::param::FloatParam(0.5);
    this->MakeSlotAvailable(&this->m_weight_0);

    this->m_weight_1 << new megamol::core::param::FloatParam(0.5);
    this->MakeSlotAvailable(&this->m_weight_1);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &TextureCombine::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &TextureCombine::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_0_slot);

    this->m_input_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_1_slot);
}

megamol::compositing::TextureCombine::~TextureCombine() {
    this->Release();
}

bool megamol::compositing::TextureCombine::create() {

    try {
        // create shader program
        m_add_prgm = std::make_unique<GLSLComputeShader>();
        m_mult_prgm = std::make_unique<GLSLComputeShader>();

        vislib_gl::graphics::gl::ShaderSource compute_add_src;
        vislib_gl::graphics::gl::ShaderSource compute_mult_src;

        auto ssf =
            std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
        if (!ssf->MakeShaderSource("Compositing::textureAdd", compute_add_src))
            return false;
        if (!m_add_prgm->Compile(compute_add_src.Code(), compute_add_src.Count()))
            return false;
        if (!m_add_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource("Compositing::textureMultiply", compute_mult_src))
            return false;
        if (!m_mult_prgm->Compile(compute_mult_src.Code(), compute_mult_src.Count()))
            return false;
        if (!m_mult_prgm->Link())
            return false;

    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile shader (@%s): %s\n",
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
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

    if (lhs_tc == NULL)
        return false;
    if (rhs_tc0 == NULL)
        return false;
    if (rhs_tc1 == NULL)
        return false;

    if (!(*rhs_tc0)(0))
        return false;
    if (!(*rhs_tc1)(0))
        return false;

    // something has changed in the neath...
    bool something_has_changed = rhs_tc0->hasUpdate() || rhs_tc1->hasUpdate();

    if (something_has_changed) {
        ++m_version;

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

            glUniform1f(
                m_add_prgm->ParameterLocation("weight0"), this->m_weight_0.Param<core::param::FloatParam>()->Value());
            glUniform1f(
                m_add_prgm->ParameterLocation("weight1"), this->m_weight_1.Param<core::param::FloatParam>()->Value());

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            m_add_prgm->Dispatch(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

            m_add_prgm->Disable();
        } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
            m_mult_prgm->Enable();

            glActiveTexture(GL_TEXTURE0);
            src0_tx2D->bindTexture();
            glUniform1i(m_mult_prgm->ParameterLocation("src0_tx2D"), 0);
            glActiveTexture(GL_TEXTURE1);
            src1_tx2D->bindTexture();
            glUniform1i(m_mult_prgm->ParameterLocation("src1_tx2D"), 1);

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            m_mult_prgm->Dispatch(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

            m_mult_prgm->Disable();
        }
    }

    lhs_tc->setData(m_output_texture, m_version);

    return true;
}

bool megamol::compositing::TextureCombine::getMetaDataCallback(core::Call& caller) {

    // TODO output hash?

    return true;
}

bool megamol::compositing::TextureCombine::modeCallback(core::param::ParamSlot& slot) {

    int mode = m_mode.Param<core::param::EnumParam>()->Value();

    // assao
    if (mode == 0) {
        m_weight_0.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_weight_1.Param<core::param::FloatParam>()->SetGUIVisible(true);
    }
    // naive
    else {
        m_weight_0.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_weight_1.Param<core::param::FloatParam>()->SetGUIVisible(false);
    }

    return true;
}
