#include "stdafx.h"
#include "AntiAliasing.h"

#include <array>
#include <random>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

#include "SMAAAreaTex.h"
#include "SMAASearchTex.h"

megamol::compositing::AntiAliasing::AntiAliasing() : core::Module()
    , m_version(0)
    , m_output_texture(nullptr)
    , m_output_texture_hash(0)
    , m_mode("Mode", "Sets screen space effect mode, e.g. ssao, fxaa...")
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects an optional input texture")
{
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "FXAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "SMAA");
    this->MakeSlotAvailable(&this->m_mode);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &AntiAliasing::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &AntiAliasing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);
}

megamol::compositing::AntiAliasing::~AntiAliasing() { this->Release(); }

bool megamol::compositing::AntiAliasing::create() {
    try {
        // create shader program
        m_fxaa_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_fxaa_src;
        vislib::graphics::gl::ShaderSource compute_smaa_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::fxaa", compute_fxaa_src))
            return false;
        if (!m_fxaa_prgm->Compile(compute_fxaa_src.Code(), compute_fxaa_src.Count()))
            return false;
        if (!m_fxaa_prgm->Link())
            return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa", compute_smaa_src))
            return false;
        if (!m_smaa_prgm->Compile(compute_smaa_src.Code(), compute_smaa_src.Count()))
            return false;
        if (!m_smaa_prgm->Link())
            return false;

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
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
    glowl::TextureLayout smaa_layout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_edges_tex = std::make_shared<glowl::Texture2D>("smaa_edges_tex", smaa_layout, nullptr);
    m_blend_tex = std::make_shared<glowl::Texture2D>("smaa_blend_tex", smaa_layout, nullptr);

    // lookup textures for smaa
    // TODO: check textures in nsight or similar to see if textures are correctly loaded
    // TODO: do this in here? or every frame in the corresponding if below
    glowl::TextureLayout area_layout(GL_RG8, AREATEX_WIDTH, AREATEX_HEIGHT, 1, GL_RG, GL_UNSIGNED_BYTE, 1);
    glowl::TextureLayout search_layout(GL_R8, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, GL_RED, GL_UNSIGNED_BYTE, 1);
    m_area_tex = std::make_shared<glowl::Texture2D>("smaa_area_tex", area_layout, areaTexBytes);
    m_search_tex = std::make_shared<glowl::Texture2D>("smaa_search_tex", search_layout, searchTexBytes);

    return true;
}

void megamol::compositing::AntiAliasing::release() {}

bool megamol::compositing::AntiAliasing::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_input = m_input_tex_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    
    if(call_input != NULL) { if (!(*call_input)(0)) return false; }

    bool something_has_changed =
        (call_input != NULL ? call_input->hasUpdate() : false);

    if (something_has_changed) {
        ++m_version;

        std::function<void(std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt)>
            setupOutputTexture = [](std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt) {
                // set output texture size to primary input texture
                std::array<float, 2> texture_res = {
                    static_cast<float>(src->getWidth()), static_cast<float>(src->getHeight())};

                if (tgt->getWidth() != std::get<0>(texture_res) || tgt->getHeight() != std::get<1>(texture_res)) {
                    glowl::TextureLayout tx_layout(
                        GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_RGBA, GL_HALF_FLOAT, 1);
                    tgt->reload(tx_layout, nullptr);
                }
            };

        auto input_tx2D = call_input->getData();

        // fxaa
        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
            if (call_input == NULL)
                return false;

            setupOutputTexture(input_tx2D, m_output_texture);

            m_fxaa_prgm->Enable();

            glActiveTexture(GL_TEXTURE0);
            input_tx2D->bindTexture();
            glUniform1i(m_fxaa_prgm->ParameterLocation("src_tx2D"), 0);

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            m_fxaa_prgm->Dispatch(static_cast<int>(std::ceil(m_output_texture->getWidth() / 8.0f)),
                static_cast<int>(std::ceil(m_output_texture->getHeight() / 8.0f)), 1);

            m_fxaa_prgm->Disable();
        }
        // smaa
        else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
            GLubyte col[4] = { 0, 0, 0, 0 };
            m_edges_tex->clearTexImage(col);
            m_blend_tex->clearTexImage(col);
        }
    }

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}

bool megamol::compositing::AntiAliasing::getMetaDataCallback(core::Call& caller) { return true; }
