#include "../include/mmcore/DeferredShading.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/graphics/gl/ShaderSource.h"

megamol::core::DeferredShading::DeferredShading() 
    : Renderer3DModule_2()
    , m_GBuffer(nullptr)
    , m_deferred_shading_prgm(nullptr)
    , m_lights_buffer(nullptr)
    , m_btf_filename_slot("BTF filename", "The name of the btf file to load") 
{
    this->m_btf_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_btf_filename_slot);
}

megamol::core::DeferredShading::~DeferredShading() { this->Release(); }

bool megamol::core::DeferredShading::create() {
    return true; 
}

void megamol::core::DeferredShading::release() { m_GBuffer.reset(); }

bool megamol::core::DeferredShading::GetExtents(core::view::CallRender3D_2& call) { 
    return true;
}

bool megamol::core::DeferredShading::Render(core::view::CallRender3D_2& call) { 

    megamol::core::view::CallRender3D_2* cr = &call; // dynamic_cast<core::view::CallRender3D_2*>(&call);
    if (cr == NULL) return false;

    // obtain camera information
    core::view::Camera_2 cam(cr->GetCamera());
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type view_tmp, proj_tmp;
    cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
    glm::mat4 view_mx = view_tmp;
    glm::mat4 proj_mx = proj_tmp;

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //m_GBuffer->bindToRead(0);
    //glBlitFramebuffer(0, 0, m_GBuffer->getWidth(), m_GBuffer->getHeight(), 0, 0, m_GBuffer->getWidth(),
    //    m_GBuffer->getHeight(), GL_COLOR_BUFFER_BIT, GL_NEAREST);

    if (m_deferred_shading_prgm == nullptr) {
        m_deferred_shading_prgm = std::make_unique<GLSLShader>();

        auto vislib_filename = m_btf_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.PeekBuffer());

        vislib::graphics::gl::ShaderSource vert_shader_src;
        vislib::graphics::gl::ShaderSource frag_shader_src;
        // TODO get rid of vislib StringA...
        vislib::StringA shader_base_name(filename.c_str());

        auto vertShaderName = shader_base_name + "::vertex";
        auto fragShaderName = shader_base_name + "::fragment";

        this->instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
        this->instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);

        try {
            m_deferred_shading_prgm->Create(
                vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
        } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
                shader_base_name.PeekBuffer(),
                vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
                ce.GetMsgA());
            // return false;
        } catch (vislib::Exception e) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s:\n%s\n",
                shader_base_name.PeekBuffer(), e.GetMsgA());
            // return false;
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown exception\n", shader_base_name.PeekBuffer());
            // return false;
        }
    }

    if (m_lights_buffer == nullptr)
    {
        m_lights_buffer = std::make_unique<glowl::BufferObject>(
            GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    }

    auto light_update = this->GetLights();
    if (light_update)
    {
        struct LightParams {
            float x, y, z, intensity;
        };

        auto light_cnt = lightMap.size();

        std::vector<LightParams> lights;
        lights.reserve(light_cnt);

        for (const auto element : this->lightMap) {
            auto light = element.second;
            lights.push_back({light.pl_position[0], light.pl_position[1], light.pl_position[2], light.lightIntensity});
        }

        m_lights_buffer->rebuffer(lights);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (m_deferred_shading_prgm != nullptr && m_lights_buffer != nullptr)
    {
        m_deferred_shading_prgm->Enable();

        m_lights_buffer->bind(1);
        glUniform1i(m_deferred_shading_prgm->ParameterLocation("light_cnt"), 1);

        glActiveTexture(GL_TEXTURE0);
        m_GBuffer->bindColorbuffer(0);
        glUniform1i(m_deferred_shading_prgm->ParameterLocation("albedo_tx2D"), 0);

        glActiveTexture(GL_TEXTURE1);
        m_GBuffer->bindColorbuffer(1);
        glUniform1i(m_deferred_shading_prgm->ParameterLocation("normal_tx2D"), 1);

        glActiveTexture(GL_TEXTURE2);
        m_GBuffer->bindColorbuffer(2);
        glUniform1i(m_deferred_shading_prgm->ParameterLocation("depth_tx2D"), 2);

        auto inv_view_mx = glm::inverse(view_mx);
        auto inv_proj_mx = glm::inverse(proj_mx);
        glUniformMatrix4fv(
            m_deferred_shading_prgm->ParameterLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(inv_view_mx));
        glUniformMatrix4fv(
            m_deferred_shading_prgm->ParameterLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(inv_proj_mx));

        glDrawArrays(GL_TRIANGLES, 0, 6);

        m_deferred_shading_prgm->Disable();
    }

    return true; 
}

void megamol::core::DeferredShading::PreRender(core::view::CallRender3D_2& call) {

    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    if (m_GBuffer == nullptr) {
        m_GBuffer = std::make_unique<glowl::FramebufferObject>(viewport[2], viewport[3], true);
        m_GBuffer->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT); // surface albedo
        m_GBuffer->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT); // normals
        m_GBuffer->createColorAttachment(GL_R32F, GL_RED, GL_FLOAT); // clip space depth
    }
    else if (m_GBuffer->getWidth() != viewport[2] || m_GBuffer->getHeight() != viewport[3]) {
        m_GBuffer->resize(viewport[2], viewport[3]);
    }

    m_GBuffer->bind();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
