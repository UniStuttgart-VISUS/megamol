#include "stdafx.h"
#include "TextureDepthCompositing.h"

#include <array>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::TextureDepthCompositing::TextureDepthCompositing()
    : core::Module()
    , m_output_texture(nullptr)
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_0_slot("InputTexture0", "Connects the primary input texture that is also used the set the output texture size")
    , m_input_tex_1_slot("InputTexture1", "Connects the secondary input texture") 
    , m_depth_tex_0_slot(
          "DepthTexture0", "Connects the primary depth texture")
    , m_depth_tex_1_slot("DepthTexture1", "Connects the secondary depth texture") {
    
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetData", &TextureDepthCompositing::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &TextureDepthCompositing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_0_slot);

    this->m_input_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_1_slot);

    this->m_depth_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_0_slot);

    this->m_depth_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_1_slot);
}

megamol::compositing::TextureDepthCompositing::~TextureDepthCompositing() { this->Release(); }

bool megamol::compositing::TextureDepthCompositing::create() {

    try {
        // create shader program
        m_depthComp_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::textureDepthCompositing", compute_src))
            return false;
        if (!m_depthComp_prgm->Compile(compute_src.Code(), compute_src.Count())) return false;
        if (!m_depthComp_prgm->Link()) return false;


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

void megamol::compositing::TextureDepthCompositing::release() {}

bool megamol::compositing::TextureDepthCompositing::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto rhs_tc0 = m_input_tex_0_slot.CallAs<CallTexture2D>();
    auto rhs_tc1 = m_input_tex_1_slot.CallAs<CallTexture2D>();

    auto rhs_dtc0 = m_depth_tex_0_slot.CallAs<CallTexture2D>();
    auto rhs_dtc1 = m_depth_tex_1_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    if (rhs_tc0 == NULL) return false;
    if (rhs_tc1 == NULL) return false;
    if (rhs_dtc0 == NULL) return false;
    if (rhs_dtc1 == NULL) return false;

    if (!(*rhs_tc0)(0)) return false;
    if (!(*rhs_tc1)(0)) return false;

    if (!(*rhs_dtc0)(0)) return false;
    if (!(*rhs_dtc1)(0)) return false;

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

    auto depth0_tx2D = rhs_dtc0->getData();
    auto depth1_tx2D = rhs_dtc1->getData();

    m_depthComp_prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    src0_tx2D->bindTexture();
    glUniform1i(m_depthComp_prgm->ParameterLocation("src0_tx2D"), 0);
    glActiveTexture(GL_TEXTURE1);
    src1_tx2D->bindTexture();
    glUniform1i(m_depthComp_prgm->ParameterLocation("src1_tx2D"), 1);
    glActiveTexture(GL_TEXTURE2);
    depth0_tx2D->bindTexture();
    glUniform1i(m_depthComp_prgm->ParameterLocation("depth0_tx2D"), 2);
    glActiveTexture(GL_TEXTURE3);
    depth1_tx2D->bindTexture();
    glUniform1i(m_depthComp_prgm->ParameterLocation("depth1_tx2D"), 3);

    m_output_texture->bindImage(0, GL_WRITE_ONLY);

    m_depthComp_prgm->Dispatch(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
        static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

    m_depthComp_prgm->Disable();

    return true;
}

bool megamol::compositing::TextureDepthCompositing::getMetaDataCallback(core::Call& caller) {

    // TODO output hash?

    return true;
}
