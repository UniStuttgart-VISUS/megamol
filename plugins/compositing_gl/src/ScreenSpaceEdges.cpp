#include "ScreenSpaceEdges.h"

#include <array>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

megamol::compositing_gl::ScreenSpaceEdges::ScreenSpaceEdges()
        : core::Module()
        , m_version(0)
        , m_output_texture(nullptr)
        , m_depth_threshold("DepthThreshold", "Threshold for drawing depth discontinuity (world space depth) as edge.")
        , m_normal_threshold("NormalThreshold", "Threshold for drawing normal discontinuity (dot product) as edge.")
        , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
        , m_depth_tex_slot("Depth", "Connects the depth texture that")
        , m_normal_tex_slot("Normal", "Connects the normal texture")
        , camera_slot_("Camera", "Connects a (copy of) camera state") {
    this->m_depth_threshold << new megamol::core::param::FloatParam(0.5);
    this->MakeSlotAvailable(&this->m_depth_threshold);

    this->m_normal_threshold << new megamol::core::param::FloatParam(0.5);
    this->MakeSlotAvailable(&this->m_normal_threshold);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &ScreenSpaceEdges::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &ScreenSpaceEdges::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_depth_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_slot);

    this->m_normal_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_normal_tex_slot);

    this->camera_slot_.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->camera_slot_);
}

megamol::compositing_gl::ScreenSpaceEdges::~ScreenSpaceEdges() {
    this->Release();
}

bool megamol::compositing_gl::ScreenSpaceEdges::create() {

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        m_edge_outline_prgm = core::utility::make_glowl_shader(
            "compositing_gl_edgeOutline", shader_options, "compositing_gl/edge_outline.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("EdgeOutline: " + std::string(e.what())).c_str());
        return false;
    }

    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("EdgeOutline_output", tx_layout, nullptr);

    return true;
}

void megamol::compositing_gl::ScreenSpaceEdges::release() {}

bool megamol::compositing_gl::ScreenSpaceEdges::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto rhs_tc0 = m_depth_tex_slot.CallAs<CallTexture2D>();
    auto rhs_tc1 = m_normal_tex_slot.CallAs<CallTexture2D>();
    auto rhs_cam = camera_slot_.CallAs<CallCamera>();

    if (lhs_tc == NULL)
        return false;
    if (rhs_tc0 == NULL)
        return false;
    if (rhs_tc1 == NULL)
        return false;
    if (rhs_cam == NULL)
        return false;

    if (!(*rhs_tc0)(0))
        return false;
    if (!(*rhs_tc1)(0))
        return false;
    if (!(*rhs_cam)(0))
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

        // obtain camera information
        core::view::Camera cam = rhs_cam->getData();
        glm::mat4 viewMx = cam.getViewMatrix();
        glm::mat4 projMx = cam.getProjectionMatrix();

        m_edge_outline_prgm->use();

        glActiveTexture(GL_TEXTURE0);
        src0_tx2D->bindTexture();
        glUniform1i(m_edge_outline_prgm->getUniformLocation("depth_tx2D"), 0);
        glActiveTexture(GL_TEXTURE1);
        src1_tx2D->bindTexture();
        glUniform1i(m_edge_outline_prgm->getUniformLocation("normal_tx2D"), 1);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);

        auto invViewMx = glm::inverse(viewMx);
        auto invProjMx = glm::inverse(projMx);
        glUniformMatrix4fv(
            m_edge_outline_prgm->getUniformLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(invViewMx));
        glUniformMatrix4fv(
            m_edge_outline_prgm->getUniformLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(invProjMx));

        glUniformMatrix4fv(m_edge_outline_prgm->getUniformLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(viewMx));
        glUniformMatrix4fv(m_edge_outline_prgm->getUniformLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(projMx));

        m_edge_outline_prgm->setUniform(
            "depth_threshold", this->m_depth_threshold.Param<core::param::FloatParam>()->Value());
        m_edge_outline_prgm->setUniform(
            "normal_threshold", this->m_normal_threshold.Param<core::param::FloatParam>()->Value());

        glDispatchCompute(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
            static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

        glUseProgram(0);
    }

    lhs_tc->setData(m_output_texture, m_version);

    return true;
}

bool megamol::compositing_gl::ScreenSpaceEdges::getMetaDataCallback(core::Call& caller) {

    // TODO output hash?

    return true;
}
