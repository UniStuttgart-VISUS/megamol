#include "SurfaceLICRenderer.h"

#include "mmcore/Call.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ScaledBoundingBoxes.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/TransferFunction.h"

#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/utility/log/Log.h"

#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>

namespace megamol {
namespace astro {

SurfaceLICRenderer::SurfaceLICRenderer()
    : input_renderer("input_renderer", "Renderer producing the surface and depth used for drawing the LIC upon")
    , input_velocities("input_velocities", "Grid with velocities")
    , input_transfer_function(
          "m_input_transfer_function", "Transfer function to color the LIC according to the velocity magnitude")
    , arc_length("arc_length", "Length of the streamlines relative to the domain size")
    , num_advections("num_advections", "Number of advections for reaching the desired arc length")
    , epsilon("epsilon", "Threshold for detecting coherent structures")
    , noise_bands("noise_bands", "Number of noise bands for LOD noise")
    , noise_scale("noise_scale", "Noise scalar for fine-tuning LOD noise")
    , coloring("coloring", "Different options on velocity coloring")
    , ka("lighting::ka", "Ambient part for Phong lighting")
    , kd("lighting::kd", "Diffuse part for Phong lighting")
    , ks("lighting::ks", "Specular part for Phong lighting")
    , shininess("lighting::shininess", "Shininess for Phong lighting")
    , ambient_color("lighting::ambient color", "Ambient color")
    , specular_color("lighting::specular color", "Specular color")
    , light_color("lighting::light color", "Light color")
    , hash(-1) {

    this->input_renderer.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->input_renderer);

    this->input_velocities.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->input_velocities);

    this->input_transfer_function.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->input_transfer_function);

    this->arc_length << new core::param::FloatParam(0.03f);
    this->MakeSlotAvailable(&this->arc_length);

    this->num_advections << new core::param::IntParam(100, 1);
    this->MakeSlotAvailable(&this->num_advections);

    this->noise_bands << new core::param::IntParam(2, 1);
    this->MakeSlotAvailable(&this->noise_bands);

    this->noise_scale << new core::param::FloatParam(5.0f);
    this->MakeSlotAvailable(&this->noise_scale);

    this->epsilon << new core::param::FloatParam(0.03f);
    this->MakeSlotAvailable(&this->epsilon);

    this->coloring << new core::param::EnumParam(0);
    this->coloring.Param<core::param::EnumParam>()->SetTypePair(0, "Original");
    this->coloring.Param<core::param::EnumParam>()->SetTypePair(1, "Projected");
    this->coloring.Param<core::param::EnumParam>()->SetTypePair(2, "Difference");
    this->MakeSlotAvailable(&this->coloring);

    this->ka << new core::param::FloatParam(0.2f, 0.0f);
    this->MakeSlotAvailable(&this->ka);

    this->kd << new core::param::FloatParam(0.6f, 0.0f);
    this->MakeSlotAvailable(&this->kd);

    this->ks << new core::param::FloatParam(0.2f, 0.0f);
    this->MakeSlotAvailable(&this->ks);

    this->shininess << new core::param::FloatParam(10.0f, 0.0f);
    this->MakeSlotAvailable(&this->shininess);

    this->ambient_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->ambient_color);

    this->specular_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->specular_color);

    this->light_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->light_color);
}

SurfaceLICRenderer::~SurfaceLICRenderer() { this->Release(); }

bool SurfaceLICRenderer::create() {
    try {
        // create shader program
        vislib::graphics::gl::ShaderSource precompute_shader_src;
        vislib::graphics::gl::ShaderSource compute_shader_src;
        vislib::graphics::gl::ShaderSource vertex_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource(
                "SurfaceLICRenderer::precompute", precompute_shader_src))
            return false;
        if (!this->pre_compute_shdr.Compile(precompute_shader_src.Code(), precompute_shader_src.Count())) return false;
        if (!this->pre_compute_shdr.Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("SurfaceLICRenderer::compute", compute_shader_src))
            return false;
        if (!this->lic_compute_shdr.Compile(compute_shader_src.Code(), compute_shader_src.Count())) return false;
        if (!this->lic_compute_shdr.Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::vert", vertex_shader_src))
            return false;
        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::frag", fragment_shader_src))
            return false;
        if (!this->render_to_framebuffer_shdr.Compile(vertex_shader_src.Code(), vertex_shader_src.Count(),
                fragment_shader_src.Code(), fragment_shader_src.Count()))
            return false;
        if (!this->render_to_framebuffer_shdr.Link()) return false;
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile shader: Unknown exception\n");
        return false;
    }
}

void SurfaceLICRenderer::release() {
}

bool SurfaceLICRenderer::GetExtents(core::Call& call) {
    auto cr = dynamic_cast<core::view::CallRender3D*>(&call);
    auto ci = this->input_renderer.CallAs<core::view::CallRender3D>();
    auto cd = this->input_velocities.CallAs<core::misc::VolumetricDataCall>();

    if (cr == nullptr) return false;
    if (ci == nullptr) return false;
    if (cd == nullptr) return false;

    int const req_frame = static_cast<int>(cr->Time());

    ci->SetTime(req_frame);
    cd->SetFrameID(req_frame, true);

    *ci = *cr;

    if (!(*ci)(core::view::CallRender3D::FnGetExtents)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;

    cr->SetTimeFramesCount(cd->FrameCount());
    cr->AccessBoundingBoxes() =
        core::utility::combineAndMagicScaleBoundingBoxes({cd->GetBoundingBoxes(), ci->GetBoundingBoxes()});

    *cr = *ci;

    return true;
}

bool SurfaceLICRenderer::Render(core::Call& call) {
    auto cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    const auto req_frame = cr->Time();

    // Get input rendering
    auto ci = this->input_renderer.CallAs<core::view::CallRender3D>();
    if (ci == nullptr) return false;

    ci->SetTime(req_frame);
    ci->SetCameraParameters(cr->GetCameraParameters());

    if (this->fbo.GetWidth() != ci->GetViewport().Width() || this->fbo.GetHeight() != ci->GetViewport().Height()) {
        if (this->fbo.IsValid()) this->fbo.Release();

        std::array<vislib::graphics::gl::FramebufferObject::ColourAttachParams, 2> cap;
        cap[0].internalFormat = GL_RGBA8;
        cap[0].format = GL_RGBA;
        cap[0].type = GL_UNSIGNED_BYTE;
        cap[1].internalFormat = GL_RGBA32F;
        cap[1].format = GL_RGBA;
        cap[1].type = GL_FLOAT;

        vislib::graphics::gl::FramebufferObject::DepthAttachParams dap;
        dap.format = GL_DEPTH_COMPONENT24;
        dap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE;

        vislib::graphics::gl::FramebufferObject::StencilAttachParams sap;
        sap.format = GL_STENCIL_INDEX;
        sap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;

        this->fbo.Create(ci->GetViewport().Width(), ci->GetViewport().Height(), cap.size(), cap.data(), dap, sap);
    }

    this->fbo.Enable();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!(*ci)(core::view::CallRender3D::FnRender)) return false;
    cr->SetTimeFramesCount(ci->TimeFramesCount());

    this->fbo.Disable();

    // Get input velocities
    auto cd = this->input_velocities.CallAs<core::misc::VolumetricDataCall>();
    if (cd == nullptr) return false;

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

    cd->SetFrameID(req_frame, true);

    if (cd->GetComponents() != 3) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Input velocities must be vectors with 3 components");
        return false;
    }

    if (this->velocity_texture == nullptr || this->velocity_texture->getWidth() != cd->GetResolution(0) ||
        this->velocity_texture->getHeight() != cd->GetResolution(1) ||
        this->velocity_texture->getDepth() != cd->GetResolution(2) || this->hash != cd->DataHash()) {

        this->hash = cd->DataHash();

        glowl::TextureLayout velocity_layout(GL_RGB32F, cd->GetResolution(0), cd->GetResolution(1),
            cd->GetResolution(2), GL_RGB, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->velocity_texture = std::make_unique<glowl::Texture3D>("velocity_texture", velocity_layout, cd->GetData());
    }

    // Get input transfer function
    auto ct = this->input_transfer_function.CallAs<core::view::CallGetTransferFunction>();

    GLuint tf_texture = 0;

    if (ct != nullptr && (*ct)()) {
        tf_texture = ct->OpenGLTexture();
    }

    // Create velocity target texture
    if (this->velocity_target == nullptr || this->velocity_target->getWidth() != cr->GetViewport().Width() ||
        this->velocity_target->getHeight() != cr->GetViewport().Height()) {

        glowl::TextureLayout velocity_tgt_layout(GL_RGBA32F, cr->GetViewport().Width(), cr->GetViewport().Height(), 1,
            GL_RGBA, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->velocity_target = std::make_unique<glowl::Texture2D>("velocity_target", velocity_tgt_layout, nullptr);
    }

    // Create render target texture
    if (this->render_target == nullptr || this->render_target->getWidth() != cr->GetViewport().Width() ||
        this->render_target->getHeight() != cr->GetViewport().Height()) {

        glowl::TextureLayout render_tgt_layout(GL_RGBA8, cr->GetViewport().Width(), cr->GetViewport().Height(), 1,
            GL_RGBA, GL_UNSIGNED_BYTE, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->render_target = std::make_unique<glowl::Texture2D>("render_target", render_tgt_layout, nullptr);
    }

    // Create noise texture
    if (this->noise_texture == nullptr) {
        glowl::TextureLayout noise_layout(GL_R32F, 64, 64, 64, GL_RED, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_REPEAT}, {GL_TEXTURE_WRAP_T, GL_REPEAT}, {GL_TEXTURE_WRAP_R, GL_REPEAT},
                {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->noise.resize(64 * 64 * 64);

        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 engine(seed);
        //std::normal_distribution<float> normal_dist(0.0f, 0.5f);
        std::uniform_real_distribution<float> normal_dist(0.0f, 1.0f);

        auto random = [&engine, &normal_dist](
                          float& value) { value = std::min(std::max(normal_dist(engine), 0.0f), 1.0f); };

        std::for_each(this->noise.begin(), this->noise.end(), random);

        this->noise_texture = std::make_unique<glowl::Texture3D>("noise_texture", noise_layout, this->noise.data());
    }

    // Get camera
    core::utility::glMagicScale scaling;
    scaling.apply(cr->GetBoundingBoxes());

    std::array<GLfloat, 16> mv_matrix, proj_matrix;
    glGetFloatv(GL_MODELVIEW_MATRIX, mv_matrix.data());
    glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix.data());

    const auto cam_near = cr->GetCameraParameters()->NearClip();
    const auto cam_far = cr->GetCameraParameters()->FarClip();

    const std::array<float, 2> rt_resolution{
        static_cast<float>(this->render_target->getWidth()), static_cast<float>(this->render_target->getHeight())};

    const std::array<float, 3> origin{cr->GetBoundingBoxes().ObjectSpaceBBox().Left(),
        cr->GetBoundingBoxes().ObjectSpaceBBox().Bottom(), cr->GetBoundingBoxes().ObjectSpaceBBox().Back()};
    const std::array<float, 3> resolution{cr->GetBoundingBoxes().ObjectSpaceBBox().Width(),
        cr->GetBoundingBoxes().ObjectSpaceBBox().Height(), cr->GetBoundingBoxes().ObjectSpaceBBox().Depth()};

    std::array<float, 4> light = {0.0f, 0.0f, 1.0f, 1.0f};
    glGetLightfv(GL_LIGHT0, GL_POSITION, light.data());

    // Transform velocities to 2D in a pre-computation step
    this->pre_compute_shdr.Enable();

    glUniformMatrix4fv(this->pre_compute_shdr.ParameterLocation("view_mx"), 1, GL_FALSE, mv_matrix.data());
    glUniformMatrix4fv(this->pre_compute_shdr.ParameterLocation("proj_mx"), 1, GL_FALSE, proj_matrix.data());

    glUniform2fv(this->pre_compute_shdr.ParameterLocation("rt_resolution"), 1, rt_resolution.data());

    glUniform3fv(this->pre_compute_shdr.ParameterLocation("origin"), 1, origin.data());
    glUniform3fv(this->pre_compute_shdr.ParameterLocation("resolution"), 1, resolution.data());

    glActiveTexture(GL_TEXTURE0);
    this->fbo.BindDepthTexture();
    glUniform1i(this->pre_compute_shdr.ParameterLocation("depth_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo.BindColourTexture(1);
    glUniform1i(this->pre_compute_shdr.ParameterLocation("normal_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    this->velocity_texture->bindTexture();
    glUniform1i(this->pre_compute_shdr.ParameterLocation("velocity_tx3D"), 2);

    this->velocity_target->bindImage(0, GL_WRITE_ONLY);

    this->pre_compute_shdr.Dispatch(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    this->pre_compute_shdr.Disable();

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // Compute surface LIC
    this->lic_compute_shdr.Enable();

    glUniformMatrix4fv(this->lic_compute_shdr.ParameterLocation("view_mx"), 1, GL_FALSE, mv_matrix.data());
    glUniformMatrix4fv(this->lic_compute_shdr.ParameterLocation("proj_mx"), 1, GL_FALSE, proj_matrix.data());

    glUniform1f(this->lic_compute_shdr.ParameterLocation("cam_near"), cam_near);
    glUniform1f(this->lic_compute_shdr.ParameterLocation("cam_far"), cam_far);

    glUniform2fv(this->lic_compute_shdr.ParameterLocation("rt_resolution"), 1, rt_resolution.data());

    glUniform3fv(this->lic_compute_shdr.ParameterLocation("origin"), 1, origin.data());
    glUniform3fv(this->lic_compute_shdr.ParameterLocation("resolution"), 1, resolution.data());

    glUniform1i(this->lic_compute_shdr.ParameterLocation("noise_bands"),
        this->noise_bands.Param<core::param::IntParam>()->Value());
    glUniform1f(this->lic_compute_shdr.ParameterLocation("noise_scale"),
        this->noise_scale.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr.ParameterLocation("arc_length"),
        this->arc_length.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->lic_compute_shdr.ParameterLocation("num_advections"),
        this->num_advections.Param<core::param::IntParam>()->Value());
    glUniform1f(
        this->lic_compute_shdr.ParameterLocation("epsilon"), this->epsilon.Param<core::param::FloatParam>()->Value());

    glUniform1i(
        this->lic_compute_shdr.ParameterLocation("coloring"), this->coloring.Param<core::param::EnumParam>()->Value());

    glUniform1f(this->lic_compute_shdr.ParameterLocation("max_magnitude"),
        static_cast<float>(cd->GetMetadata()->MaxValues[0]));

    glUniform1f(this->lic_compute_shdr.ParameterLocation("ka"), this->ka.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr.ParameterLocation("kd"), this->kd.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr.ParameterLocation("ks"), this->ks.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr.ParameterLocation("shininess"),
        this->shininess.Param<core::param::FloatParam>()->Value());
    glUniform3fv(this->lic_compute_shdr.ParameterLocation("light"), 1, light.data());
    glUniform3fv(this->lic_compute_shdr.ParameterLocation("ambient_col"), 1,
        this->ambient_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(this->lic_compute_shdr.ParameterLocation("specular_col"), 1,
        this->specular_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(this->lic_compute_shdr.ParameterLocation("light_col"), 1,
        this->light_color.Param<core::param::ColorParam>()->Value().data());

    glActiveTexture(GL_TEXTURE0);
    this->fbo.BindDepthTexture();
    glUniform1i(this->lic_compute_shdr.ParameterLocation("depth_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->velocity_target->bindTexture();
    glUniform1i(this->lic_compute_shdr.ParameterLocation("velocity_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    this->fbo.BindColourTexture(1);
    glUniform1i(this->lic_compute_shdr.ParameterLocation("normal_tx2D"), 2);

    glActiveTexture(GL_TEXTURE3);
    this->noise_texture->bindTexture();
    glUniform1i(this->lic_compute_shdr.ParameterLocation("noise_tx3D"), 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_1D, tf_texture);
    glUniform1i(this->lic_compute_shdr.ParameterLocation("tf_tx1D"), 4);

    this->render_target->bindImage(0, GL_WRITE_ONLY);

    this->lic_compute_shdr.Dispatch(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_1D, 0);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, 0);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    this->lic_compute_shdr.Disable();

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // Render to framebuffer
    bool state_depth_test = glIsEnabled(GL_DEPTH_TEST);
    bool state_blend = glIsEnabled(GL_BLEND);

    if (!state_depth_test) glEnable(GL_DEPTH_TEST);
    if (state_blend) glDisable(GL_BLEND);

    this->render_to_framebuffer_shdr.Enable();

    glActiveTexture(GL_TEXTURE0);
    this->render_target->bindTexture();
    glUniform1i(this->render_to_framebuffer_shdr.ParameterLocation("src_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo.BindDepthTexture();
    glUniform1i(this->render_to_framebuffer_shdr.ParameterLocation("depth_tx2D"), 1);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    this->render_to_framebuffer_shdr.Disable();

    if (state_blend) glEnable(GL_BLEND);
    if (!state_depth_test) glDisable(GL_DEPTH_TEST);
}

} // namespace astro
} // namespace megamol
