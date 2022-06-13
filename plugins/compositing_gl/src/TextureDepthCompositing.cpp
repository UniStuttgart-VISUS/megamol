#include "TextureDepthCompositing.h"

#include <array>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"

#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"

megamol::compositing::TextureDepthCompositing::TextureDepthCompositing()
        : core::Module()
        , m_version(0)
        , m_depthComp_prgm(nullptr)
        , m_output_texture(nullptr)
        , m_output_depth_texture(nullptr)
        , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
        , m_output_depth_tex_slot("OutputDepthTexture", "Gives access to resulting output depth texture")
        , m_input_tex_0_slot(
              "InputTexture0", "Connects the primary input texture that is also used the set the output texture size")
        , m_input_tex_1_slot("InputTexture1", "Connects the secondary input texture")
        , m_depth_tex_0_slot("DepthTexture0", "Connects the primary depth texture")
        , m_depth_tex_1_slot("DepthTexture1", "Connects the secondary depth texture") {

    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetData", &TextureDepthCompositing::getOutputImageCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &TextureDepthCompositing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_output_depth_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetData", &TextureDepthCompositing::getDepthImageCallback);
    this->m_output_depth_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &TextureDepthCompositing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_depth_tex_slot);

    this->m_input_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_0_slot);

    this->m_input_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_1_slot);

    this->m_depth_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_0_slot);

    this->m_depth_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_1_slot);
}

megamol::compositing::TextureDepthCompositing::~TextureDepthCompositing() {
    this->Release();
}

bool megamol::compositing::TextureDepthCompositing::create() {

    try {
        // create shader program
        vislib_gl::graphics::gl::ShaderSource compute_src;

        auto ssf =
            std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
        if (!ssf->MakeShaderSource("Compositing::textureDepthCompositing", compute_src)) {
            return false;
        }

        std::string compute_shader_src(compute_src.WholeCode(), (compute_src.WholeCode()).Length());

        std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs;
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Compute, compute_shader_src});

        m_depthComp_prgm = std::make_unique<glowl::GLSLProgram>(shader_srcs);
        m_depthComp_prgm->setDebugLabel(
            "Compositing::textureDepthCompositing"); //TODO debug label not set in time for catch...

    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n",
            m_depthComp_prgm->getDebugLabel().c_str(), exc.what(), __FILE__, __FUNCTION__, __LINE__);
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

    glowl::TextureLayout depth_tx_layout(GL_R32F, 1, 1, 1, GL_R, GL_FLOAT, 1);
    m_output_depth_texture = std::make_shared<glowl::Texture2D>("textureDepthCombine_output", depth_tx_layout, nullptr);

    return true;
}

void megamol::compositing::TextureDepthCompositing::release() {}

bool megamol::compositing::TextureDepthCompositing::getOutputImageCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    if (lhs_tc == NULL)
        return false;

    if (!computeDepthCompositing())
        return false;

    lhs_tc->setData(m_output_texture, m_version);
    return true;
}

bool megamol::compositing::TextureDepthCompositing::getDepthImageCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    if (lhs_tc == NULL)
        return false;

    if (!computeDepthCompositing())
        return false;

    lhs_tc->setData(m_output_depth_texture, m_version);

    return true;
}

bool megamol::compositing::TextureDepthCompositing::getMetaDataCallback(core::Call& caller) {
    return true;
}

bool megamol::compositing::TextureDepthCompositing::computeDepthCompositing() {

    auto rhs_tc0 = m_input_tex_0_slot.CallAs<CallTexture2D>();
    auto rhs_tc1 = m_input_tex_1_slot.CallAs<CallTexture2D>();
    auto rhs_dtc0 = m_depth_tex_0_slot.CallAs<CallTexture2D>();
    auto rhs_dtc1 = m_depth_tex_1_slot.CallAs<CallTexture2D>();

    if (rhs_tc0 == NULL)
        return false;
    if (rhs_tc1 == NULL)
        return false;
    if (rhs_dtc0 == NULL)
        return false;
    if (rhs_dtc1 == NULL)
        return false;

    if (!(*rhs_tc0)(0))
        return false;
    if (!(*rhs_tc1)(0))
        return false;
    if (!(*rhs_dtc0)(0))
        return false;
    if (!(*rhs_dtc1)(0))
        return false;

    // something has changed in the neath...
    bool something_has_changed =
        rhs_tc0->hasUpdate() || rhs_tc1->hasUpdate() || rhs_dtc0->hasUpdate() || rhs_dtc1->hasUpdate();

    if (something_has_changed) {
        ++m_version;

        auto src0_tx2D = rhs_tc0->getData();
        auto src1_tx2D = rhs_tc1->getData();

        auto depth0_tx2D = rhs_dtc0->getData();
        auto depth1_tx2D = rhs_dtc1->getData();

        // set output texture size to primary input texture

        std::array<float, 2> texture_res = {
            static_cast<float>(src0_tx2D->getWidth()), static_cast<float>(src0_tx2D->getHeight())};

        if (m_output_texture->getWidth() != std::get<0>(texture_res) ||
            m_output_texture->getHeight() != std::get<1>(texture_res)) {
            glowl::TextureLayout tx_layout(
                GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_RGBA, GL_HALF_FLOAT, 1);
            m_output_texture->reload(tx_layout, nullptr);

            glowl::TextureLayout depth_tx_layout(
                GL_R32F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_R, GL_FLOAT, 1);
            m_output_depth_texture->reload(depth_tx_layout, nullptr);
        }

        m_depthComp_prgm->use();

        glActiveTexture(GL_TEXTURE0);
        src0_tx2D->bindTexture();
        m_depthComp_prgm->setUniform("src0_tx2D", 0);
        glActiveTexture(GL_TEXTURE1);
        src1_tx2D->bindTexture();
        m_depthComp_prgm->setUniform("src1_tx2D", 1);
        glActiveTexture(GL_TEXTURE2);
        depth0_tx2D->bindTexture();
        m_depthComp_prgm->setUniform("depth0_tx2D", 2);
        glActiveTexture(GL_TEXTURE3);
        depth1_tx2D->bindTexture();
        m_depthComp_prgm->setUniform("depth1_tx2D", 3);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);
        m_output_depth_texture->bindImage(1, GL_WRITE_ONLY);

        glDispatchCompute(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
            static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

        glUseProgram(0);

        glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glBindImageTexture(1, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    }

    return true;
}
