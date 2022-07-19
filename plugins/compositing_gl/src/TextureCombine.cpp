#include "TextureCombine.h"

#include <array>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

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

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        m_add_prgm = core::utility::make_glowl_shader(
            "Compositing_textureAdd", shader_options, "compositing_gl/textureAdd.comp.glsl");

        m_mult_prgm = core::utility::make_glowl_shader(
            "Compositing_textureMultiply", shader_options, "compositing_gl/textureMultiply.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("TextureCombine: " + std::string(e.what())).c_str());
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
            m_add_prgm->use();

            glActiveTexture(GL_TEXTURE0);
            src0_tx2D->bindTexture();
            glUniform1i(m_add_prgm->getUniformLocation("src0_tx2D"), 0);
            glActiveTexture(GL_TEXTURE1);
            src1_tx2D->bindTexture();
            glUniform1i(m_add_prgm->getUniformLocation("src1_tx2D"), 1);

            glUniform1f(
                m_add_prgm->getUniformLocation("weight0"), this->m_weight_0.Param<core::param::FloatParam>()->Value());
            glUniform1f(
                m_add_prgm->getUniformLocation("weight1"), this->m_weight_1.Param<core::param::FloatParam>()->Value());

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            glDispatchCompute(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

            glUseProgram(0);
        } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
            m_mult_prgm->use();

            glActiveTexture(GL_TEXTURE0);
            src0_tx2D->bindTexture();
            glUniform1i(m_mult_prgm->getUniformLocation("src0_tx2D"), 0);
            glActiveTexture(GL_TEXTURE1);
            src1_tx2D->bindTexture();
            glUniform1i(m_mult_prgm->getUniformLocation("src1_tx2D"), 1);

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            glDispatchCompute(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

            glUseProgram(0);
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
