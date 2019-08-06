#include "SurfaceLICRenderer.h"

#include "mmcore/Call.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ScaledBoundingBoxes.h"
#include "mmcore/view/AbstractCallRender3D.h"

#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace megamol {
namespace astro {

SurfaceLICRenderer::SurfaceLICRenderer()
    : m_input_renderer("input_renderer", "Renderer producing the surface and depth used for drawing the LIC upon")
    , m_input_velocities("input_velocities", "Grid with velocities")
    , color("color", "Color of the LIC")
    , stencil_size("stencil_size", "Stencil size for thicker LIC")
    , num_advections("num_advections", "Number of advection steps")
    , timestep("timestep", "Timestep for scaling the input velocities")
    , epsilon("epsilon", "Threshold for detecting coherent structures")
    , m_lic_compute_shdr(nullptr)
    , m_render_to_framebuffer_shdr(nullptr)
    , m_render_target(nullptr) {

    this->m_input_renderer.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->m_input_renderer);

    this->m_input_velocities.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->m_input_velocities);

    this->color << new core::param::ColorParam(0.41f, 0.62f, 0.31f, 1.0f);
    this->MakeSlotAvailable(&this->color);

    this->stencil_size << new core::param::IntParam(3);
    this->MakeSlotAvailable(&this->stencil_size);

    this->num_advections << new core::param::IntParam(10);
    this->MakeSlotAvailable(&this->num_advections);

    this->timestep << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->timestep);

    this->epsilon << new core::param::FloatParam(0.01f);
    this->MakeSlotAvailable(&this->epsilon);
}

SurfaceLICRenderer::~SurfaceLICRenderer() { this->Release(); }

bool SurfaceLICRenderer::create() {
    try {
        // create shader program
        this->m_lic_compute_shdr = std::make_unique<vislib::graphics::gl::GLSLComputeShader>();
        this->m_render_to_framebuffer_shdr = std::make_unique<vislib::graphics::gl::GLSLShader>();

        vislib::graphics::gl::ShaderSource compute_shader_src;
        vislib::graphics::gl::ShaderSource vertex_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("SurfaceLICRenderer::compute", compute_shader_src))
            return false;
        if (!this->m_lic_compute_shdr->Compile(compute_shader_src.Code(), compute_shader_src.Count())) return false;
        if (!this->m_lic_compute_shdr->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::vert", vertex_shader_src))
            return false;
        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::frag", fragment_shader_src))
            return false;
        if (!this->m_render_to_framebuffer_shdr->Compile(vertex_shader_src.Code(), vertex_shader_src.Count(),
                fragment_shader_src.Code(), fragment_shader_src.Count()))
            return false;
        if (!this->m_render_to_framebuffer_shdr->Link()) return false;
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

    // create render target texture
    TextureLayout render_tgt_layout(GL_RGBA8, 1920, 1080, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    this->m_render_target = std::make_unique<Texture2D>("render_target", render_tgt_layout, nullptr);

    TextureLayout velocity_layout(GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    this->m_velocity_texture = std::make_unique<Texture3D>("velocity_texture", velocity_layout, nullptr);

    TextureLayout noise_layout(GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    this->m_noise_texture = std::make_unique<Texture2D>("noise_texture", noise_layout, nullptr);
}

void SurfaceLICRenderer::release() {
    this->m_lic_compute_shdr.reset(nullptr);
    this->m_render_to_framebuffer_shdr.reset(nullptr);

    this->m_render_target.reset(nullptr);
    this->m_velocity_texture.reset(nullptr);
    this->m_noise_texture.reset(nullptr);
}

bool SurfaceLICRenderer::GetExtents(core::Call& call) {
    auto cr = dynamic_cast<core::view::CallRender3D*>(&call);
    auto ci = this->m_input_renderer.CallAs<core::view::CallRender3D>();
    auto cd = this->m_input_velocities.CallAs<core::misc::VolumetricDataCall>();

    if (cr == nullptr) return false;
    if (ci == nullptr) return false;
    if (cd == nullptr) return false;

    int const req_frame = static_cast<int>(cr->Time());

    cd->SetFrameID(req_frame);

    if (!(*ci)(core::view::CallRender3D::FnGetExtents)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;

    cr->SetTimeFramesCount(cd->FrameCount());
    cr->AccessBoundingBoxes() =
        core::utility::combineAndMagicScaleBoundingBoxes({cd->GetBoundingBoxes(), ci->GetBoundingBoxes()});

    return true;
}

bool SurfaceLICRenderer::Render(core::Call& call) {
    auto cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    // Get input rendering
    auto ci = this->m_input_renderer.CallAs<core::view::CallRender3D>();
    if (ci == nullptr) return false;

    ci->SetCameraParameters(cr->GetCameraParameters());

    if (this->fbo.IsValid()) this->fbo.Release();

    std::array<vislib::graphics::gl::FramebufferObject::ColourAttachParams, 2> cap;
    cap[0].internalFormat = GL_RGBA8;
    cap[0].format = GL_RGBA;
    cap[0].type = GL_UNSIGNED_BYTE;
    cap[1].internalFormat = GL_RGB8;
    cap[1].format = GL_RGB;
    cap[1].type = GL_UNSIGNED_BYTE;

    vislib::graphics::gl::FramebufferObject::DepthAttachParams dap;
    dap.format = GL_DEPTH_COMPONENT24;
    dap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE;

    vislib::graphics::gl::FramebufferObject::StencilAttachParams sap;
    sap.format = GL_STENCIL_INDEX;
    sap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;

    this->fbo.Create(ci->GetViewport().Width(), ci->GetViewport().Height(), cap.size(), cap.data(), dap, sap);
    this->fbo.Enable();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!(*ci)(core::view::CallRender3D::FnRender)) return false;

    this->fbo.Disable();

    // Get input velocities
    auto cd = this->m_input_velocities.CallAs<core::misc::VolumetricDataCall>();
    if (cd == nullptr) return false;

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

    if (cd->GetComponents() != 3) {
        vislib::sys::Log::DefaultLog.WriteError("Input velocities must be vectors with 3 components");
        return false;
    }

    TextureLayout velocity_layout(GL_RGB32F, cd->GetResolution(0), cd->GetResolution(1), cd->GetResolution(2), GL_RGB,
        GL_FLOAT, 3,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});

    this->m_velocity_texture->reload(velocity_layout, cd->GetData());

    // Create noise texture
    const auto screensize = ci->GetViewport().Width() * ci->GetViewport().Height();

    if (this->noise.size() != screensize) {
        TextureLayout noise_layout(GL_R32F, ci->GetViewport().Width(), ci->GetViewport().Height(), 1, GL_RED, GL_FLOAT,
            1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->noise.resize(screensize);

        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 engine(seed);
        std::normal_distribution<float> normal_dist(0.0f, 0.5f);

        auto random = [&engine, &normal_dist](
                          float& value) { value = std::min(std::max(normal_dist(engine), 0.0f), 1.0f); };

        std::for_each(this->noise.begin(), this->noise.end(), random);

        this->m_noise_texture->reload(noise_layout, this->noise.data());
    }

    // Compute surface LIC
    this->m_lic_compute_shdr->Enable();

    core::utility::glMagicScale scaling;
    scaling.apply(cr->GetBoundingBoxes());

    std::array<GLfloat, 16> mv_matrix, proj_matrix;
    glGetFloatv(GL_MODELVIEW_MATRIX, mv_matrix.data());
    glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix.data());
    glUniformMatrix4fv(this->m_lic_compute_shdr->ParameterLocation("view_mx"), 1, GL_FALSE, mv_matrix.data());
    glUniformMatrix4fv(this->m_lic_compute_shdr->ParameterLocation("proj_mx"), 1, GL_FALSE, proj_matrix.data());

    const auto cam_near = cr->GetCameraParameters()->NearClip();
    const auto cam_far = cr->GetCameraParameters()->FarClip();
    glUniform1f(this->m_lic_compute_shdr->ParameterLocation("cam_near"), cam_near);
    glUniform1f(this->m_lic_compute_shdr->ParameterLocation("cam_far"), cam_far);

    const std::array<float, 2> rt_resolution{
        static_cast<float>(this->m_render_target->getWidth()), static_cast<float>(this->m_render_target->getHeight())};
    glUniform2fv(this->m_lic_compute_shdr->ParameterLocation("rt_resolution"), 1, rt_resolution.data());

    const std::array<float, 3> origin{cr->GetBoundingBoxes().ObjectSpaceBBox().Left(),
        cr->GetBoundingBoxes().ObjectSpaceBBox().Bottom(), cr->GetBoundingBoxes().ObjectSpaceBBox().Back()};
    glUniform3fv(this->m_lic_compute_shdr->ParameterLocation("origin"), 1, origin.data());

    const std::array<float, 3> resolution{cr->GetBoundingBoxes().ObjectSpaceBBox().Width(),
        cr->GetBoundingBoxes().ObjectSpaceBBox().Height(), cr->GetBoundingBoxes().ObjectSpaceBBox().Depth()};
    glUniform3fv(this->m_lic_compute_shdr->ParameterLocation("resolution"), 1, resolution.data());

    glUniform4fv(this->m_lic_compute_shdr->ParameterLocation("lic_color"), 1,
        this->color.Param<core::param::ColorParam>()->Value().data());

    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("stencil"),
        this->stencil_size.Param<core::param::IntParam>()->Value());

    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("num_advections"),
        this->num_advections.Param<core::param::IntParam>()->Value());

    glUniform1f(this->m_lic_compute_shdr->ParameterLocation("timestep"),
        this->timestep.Param<core::param::FloatParam>()->Value());

    glUniform1f(this->m_lic_compute_shdr->ParameterLocation("epsilon"),
        this->epsilon.Param<core::param::FloatParam>()->Value());

    // ...

    glActiveTexture(GL_TEXTURE0);
    this->fbo.BindColourTexture();
    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("color_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo.BindDepthTexture();
    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("depth_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    this->m_velocity_texture->bindTexture();
    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("velocity_tx3D"), 2);

    glActiveTexture(GL_TEXTURE3);
    this->m_noise_texture->bindTexture();
    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("noise_tx2D"), 3);

    glActiveTexture(GL_TEXTURE4);
    this->fbo.BindColourTexture(1);
    glUniform1i(this->m_lic_compute_shdr->ParameterLocation("normal_tx2D"), 0);

    this->m_render_target->bindImage(0, GL_WRITE_ONLY);

    this->m_lic_compute_shdr->Dispatch(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    this->m_lic_compute_shdr->Disable();

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // Render to framebuffer
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    this->m_render_to_framebuffer_shdr->Enable();

    glActiveTexture(GL_TEXTURE0);
    this->m_render_target->bindTexture();
    glUniform1i(this->m_render_to_framebuffer_shdr->ParameterLocation("src_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo.BindDepthTexture();
    glUniform1i(this->m_render_to_framebuffer_shdr->ParameterLocation("depth_tx2D"), 1);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    this->m_render_to_framebuffer_shdr->Disable();
}

} // namespace astro
} // namespace megamol