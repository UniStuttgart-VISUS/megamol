#include "SRTest.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmstd/light/CallLight.h"
#include "mmstd/light/DistantLight.h"
#include "OpenGL_Context.h"

#ifdef MEGAMOL_USE_CUDA
#include "SphereRasterizer.h"
#endif

#include "stb/stb_image_write.h"

#ifdef USE_NVPERF
#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>
#endif

#define SCALE 0.0001f
#define __SRTEST_CON_RAS__
//#define __SRTEST_CAM_ALIGNED__

megamol::moldyn_gl::rendering::SRTest::SRTest()
        : data_in_slot_("inData", "")
        , getLightsSlot("lights", "Lights are retrieved over this slot.")
        , method_slot_("method", "")
        , upload_mode_slot_("upload mode", "")
        , enforce_upload_slot_("enforce upload", "")
        , use_con_ras_slot_("use_con_ras", "")
/*
, clip_thres_slot_("clip distance", "")*/
{
    data_in_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightsSlot);

    auto ep = new core::param::EnumParam(static_cast<method_ut>(method_e::VAO));
    ep->SetTypePair(static_cast<method_ut>(method_e::VAO), "VAO");
    ep->SetTypePair(static_cast<method_ut>(method_e::TEX), "TEX");
    ep->SetTypePair(static_cast<method_ut>(method_e::COPY), "COPY");
    ep->SetTypePair(static_cast<method_ut>(method_e::COPY_VERT), "COPY_VERT");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO), "SSBO");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_GEO), "SSBO_GEO");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_VERT), "SSBO_VERT");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_QUAD), "SSBO_QUAD");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_STRIP), "SSBO_STRIP");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_MUZIC), "SSBO_MUZIC");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH), "MESH");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_ALTN), "MESH_ALTN");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_GEO), "MESH_GEO");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_GEO_TASK), "MESH_GEO_TASK");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_GEO_ALTN), "MESH_GEO_ALTN");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_GEO_CAM), "MESH_GEO_CAM");
    ep->SetTypePair(static_cast<method_ut>(method_e::THRUST), "THRUST");
    method_slot_ << ep;
    MakeSlotAvailable(&method_slot_);

    ep = new core::param::EnumParam(static_cast<upload_mode_ut>(upload_mode::FULL_SEP));
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::FULL_SEP), "FULL_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::POS_COL_SEP), "POS_COL_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::VEC3_SEP), "VEC3_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::NO_SEP), "NO_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::BUFFER_ARRAY), "BUFFER_ARRAY");
    upload_mode_slot_ << ep;
    MakeSlotAvailable(&upload_mode_slot_);

    enforce_upload_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&enforce_upload_slot_);

    use_con_ras_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&use_con_ras_slot_);

    /*clip_thres_slot_ << new core::param::FloatParam(0.00001f, 0.0f);
    MakeSlotAvailable(&clip_thres_slot_);*/
}


megamol::moldyn_gl::rendering::SRTest::~SRTest() {
    this->Release();
}


bool megamol::moldyn_gl::rendering::SRTest::create_shaders() {
    try {
        auto base_options =
            core::utility::make_path_shader_options(frontend_resources.get<frontend_resources::RuntimeConfig>());
        auto shdr_vao_options = base_options;
        shdr_vao_options.addDefinition("__SRTEST_VAO__");
        auto shdr_tex_options = base_options;
        shdr_tex_options.addDefinition("__SRTEST_TEX__");
        auto shdr_ssbo_options = base_options;
        shdr_ssbo_options.addDefinition("__SRTEST_SSBO__");
        auto shdr_ssbo_vert_options = base_options;
        shdr_ssbo_vert_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_vert_options.addDefinition("BASE_IDX", VERT_BASE_IDX);
        shdr_ssbo_vert_options.addDefinition("INV_IDX", VERT_INV_IDX);
        shdr_ssbo_vert_options.addDefinition("BUMP_IDX", VERT_BUMP_IDX);
        auto shdr_ssbo_quads_options = base_options;
        shdr_ssbo_quads_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_quads_options.addDefinition("__SRTEST_QUAD__");
#ifdef __SRTEST_CAM_ALIGNED__
        shdr_ssbo_quads_options.addDefinition("__SRTEST_CAM_ALIGNED__");
#endif
        shdr_ssbo_quads_options.addDefinition("BASE_IDX", QUADS_BASE_IDX);
        shdr_ssbo_quads_options.addDefinition("INV_IDX", QUADS_INV_IDX);
        shdr_ssbo_quads_options.addDefinition("BUMP_IDX", QUADS_BUMP_IDX);
        auto shdr_ssbo_strip_options = base_options;
        shdr_ssbo_strip_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_strip_options.addDefinition("BASE_IDX", STRIP_BASE_IDX);
        shdr_ssbo_strip_options.addDefinition("INV_IDX", STRIP_INV_IDX);
        shdr_ssbo_strip_options.addDefinition("BUMP_IDX", STRIP_BUMP_IDX);
        auto shdr_ssbo_muzic_options = base_options;
        shdr_ssbo_muzic_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_muzic_options.addDefinition("__SRTEST_MUZIC__");
        shdr_ssbo_muzic_options.addDefinition("BASE_IDX", MUZIC_BASE_IDX);
        shdr_ssbo_muzic_options.addDefinition("INV_IDX", MUZIC_INV_IDX);
        shdr_ssbo_muzic_options.addDefinition("BUMP_IDX", MUZIC_BUMP_IDX);
        auto shdr_copy_options = base_options;
        shdr_copy_options.addDefinition("__SRTEST_SSBO__");
        shdr_copy_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
        auto shdr_copy_vert_options = base_options;
        shdr_copy_vert_options.addDefinition("__SRTEST_SSBO__");
        shdr_copy_vert_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
        shdr_copy_vert_options.addDefinition("BASE_IDX", VERT_BASE_IDX);
        shdr_copy_vert_options.addDefinition("INV_IDX", VERT_INV_IDX);
        shdr_copy_vert_options.addDefinition("BUMP_IDX", VERT_BUMP_IDX);
        auto shdr_mesh_options = base_options;
        shdr_mesh_options.addDefinition("__SRTEST_MESH__");
        shdr_mesh_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto shdr_mesh_altn_options = base_options;
        shdr_mesh_altn_options.addDefinition("__SRTEST_MESH_ALTN__");
        shdr_mesh_altn_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto shdr_mesh_geo_options = base_options;
        shdr_mesh_geo_options.addDefinition("__SRTEST_MESH_GEO__");
#ifdef __SRTEST_CAM_ALIGNED__
        shdr_mesh_geo_options.addDefinition("__SRTEST_CAM_ALIGNED__");
#endif
        shdr_mesh_geo_options.addDefinition("__SRTEST_MESH_GEO__");
        shdr_mesh_geo_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto shdr_mesh_geo_altn_options = base_options;
        shdr_mesh_geo_altn_options.addDefinition("__SRTEST_MESH_GEO_ALTN__");
        shdr_mesh_geo_altn_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto mode = static_cast<upload_mode>(upload_mode_slot_.Param<core::param::EnumParam>()->Value());

        auto shdr_mesh_geo_cam_options = base_options;
        shdr_mesh_geo_cam_options.addDefinition("__SRTEST_MESH_GEO__");
        shdr_mesh_geo_cam_options.addDefinition("__SRTEST_CAM_ALIGNED__");
        shdr_mesh_geo_cam_options.addDefinition("__SRTEST_MESH_GEO__");
        shdr_mesh_geo_cam_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));


        switch (mode) {
        case upload_mode::FULL_SEP: {
            shdr_ssbo_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_ssbo_vert_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_ssbo_quads_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_ssbo_strip_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_ssbo_muzic_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_mesh_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_mesh_altn_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_mesh_geo_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_mesh_geo_altn_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
            shdr_mesh_geo_cam_options.addDefinition("__SRTEST_UPLOAD_FULL_SEP__");
        } break;
        case upload_mode::VEC3_SEP: {
            shdr_ssbo_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_ssbo_vert_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_ssbo_quads_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_ssbo_strip_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_ssbo_muzic_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_mesh_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_mesh_altn_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_mesh_geo_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_mesh_geo_altn_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
            shdr_mesh_geo_cam_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
        } break;
        case upload_mode::NO_SEP: {
            shdr_ssbo_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_ssbo_vert_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_ssbo_quads_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_ssbo_strip_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_ssbo_muzic_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_mesh_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_mesh_altn_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_mesh_geo_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_mesh_geo_altn_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
            shdr_mesh_geo_cam_options.addDefinition("__SRTEST_UPLOAD_VEC3_SEP__");
        } break;
        case upload_mode::BUFFER_ARRAY: {
            shdr_ssbo_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_ssbo_vert_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_ssbo_quads_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_ssbo_strip_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_ssbo_muzic_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_mesh_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_mesh_altn_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_mesh_geo_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_mesh_geo_altn_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
            shdr_mesh_geo_cam_options.addDefinition("__SRTEST_UPLOAD_BUFFER_ARRAY__");
        } break;
        case upload_mode::POS_COL_SEP:
        default: {
            shdr_ssbo_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_ssbo_vert_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_ssbo_quads_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_ssbo_strip_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_ssbo_muzic_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_mesh_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_mesh_altn_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_mesh_geo_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_mesh_geo_altn_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
            shdr_mesh_geo_cam_options.addDefinition("__SRTEST_UPLOAD_POS_COL_SEP__");
        }
        }

        rendering_tasks_.insert(std::make_pair(method_e::VAO, std::make_shared<vao_rt>(shdr_vao_options)));

        rendering_tasks_.insert(std::make_pair(method_e::TEX, std::make_shared<tex_rt>(shdr_tex_options)));

        rendering_tasks_.insert(std::make_pair(method_e::COPY, std::make_shared<copy_rt>(shdr_copy_options)));

        rendering_tasks_.insert(
            std::make_pair(method_e::COPY_VERT, std::make_shared<copy_vert_rt>(shdr_copy_vert_options)));

        rendering_tasks_.insert(std::make_pair(method_e::SSBO, std::make_shared<ssbo_rt>(mode, shdr_ssbo_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_GEO, std::make_shared<ssbo_geo_rt>(mode, shdr_ssbo_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_VERT, std::make_shared<ssbo_vert_rt>(mode, shdr_ssbo_vert_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_QUAD, std::make_shared<ssbo_quad_rt>(mode, shdr_ssbo_quads_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_STRIP, std::make_shared<ssbo_strip_rt>(mode, shdr_ssbo_strip_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_MUZIC, std::make_shared<ssbo_muzic_rt>(mode, shdr_ssbo_muzic_options)));

        if (mesh_shader_avail_) {
            rendering_tasks_.insert(std::make_pair(method_e::MESH, std::make_shared<mesh_rt>(mode, shdr_mesh_options)));

            rendering_tasks_.insert(
                std::make_pair(method_e::MESH_ALTN, std::make_shared<mesh_altn_rt>(mode, shdr_mesh_altn_options)));

            rendering_tasks_.insert(
                std::make_pair(method_e::MESH_GEO, std::make_shared<mesh_geo_rt>(mode, shdr_mesh_geo_options)));
            rendering_tasks_.insert(std::make_pair(
                method_e::MESH_GEO_TASK, std::make_shared<mesh_geo_task_rt>(mode, shdr_mesh_geo_options)));

            rendering_tasks_.insert(
                std::make_pair(method_e::MESH_GEO_CAM, std::make_shared<mesh_geo_rt>(mode, shdr_mesh_geo_cam_options)));

            rendering_tasks_.insert(std::make_pair(
                method_e::MESH_GEO_ALTN, std::make_shared<mesh_geo_altn_rt>(mode, shdr_mesh_geo_altn_options)));
        }
    } catch (glowl::GLSLProgramException const& e) {
        core::utility::log::Log::DefaultLog.WriteError("[SRTest] %s", e.what());
        return false;
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteError("[SRTest] Failed to create shader programs");
        return false;
    }

    return true;
}


bool megamol::moldyn_gl::rendering::SRTest::update_upload_setting() {
    rendering_tasks_.clear();

    return create_shaders();
}


bool megamol::moldyn_gl::rendering::SRTest::create() {
#ifdef MEGAMOL_USE_PROFILING
    auto& pm = const_cast<frontend_resources::PerformanceManager&>(
        frontend_resources.get<frontend_resources::PerformanceManager>());
    frontend_resources::PerformanceManager::basic_timer_config upload_timer, render_timer, compute_timer, thrust_timer;
    upload_timer.name = "upload";
    upload_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    compute_timer.name = "compute";
    compute_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    thrust_timer.name = "thrust";
    thrust_timer.api = frontend_resources::PerformanceManager::query_api::CPU;
    timing_handles_ = pm.add_timers(this, {upload_timer, render_timer, compute_timer, thrust_timer});
#endif
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isExtAvailable("GL_NV_mesh_shader")) {
        core::utility::log::Log::DefaultLog.WriteWarn("[SRTest]: GL_NV_mesh_shader extension not available");
        mesh_shader_avail_ = false;
    } else {
        core::utility::log::Log::DefaultLog.WriteInfo("[SRTest]: GL_NV_mesh_shader extension is available");
        mesh_shader_avail_ = true;
    }

    if (!create_shaders())
        return false;

    glCreateBuffers(1, &ubo_);
    glNamedBufferData(ubo_, sizeof(ubo_params_t), nullptr, GL_DYNAMIC_DRAW);

    GLint max_vert;
    GLint max_ind;
    glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &max_vert);
    glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &max_ind);

    core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] Max Vert %d; Max Ind %d", max_vert, max_ind);

    GLint max_ssbo_size;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_ssbo_size);

    core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] Max SSBO Size %d;", max_ssbo_size);

#ifdef USE_NVPERF
    std::filesystem::path profile_output =
        frontend_resources.get<frontend_resources::RuntimeConfig>().profiling_output_file;
    auto profile_path = profile_output.remove_filename();
    nv::perf::InitializeNvPerf();
    nvperf.InitializeReportGenerator();
    nvperf.SetFrameLevelRangeName("Frame");
    nvperf.SetNumNestingLevels(3);
    //nvperf.outputOptions.directoryName = ".\\nvperf";
    nvperf.outputOptions.directoryName = profile_path.string();
    clockStatus = nv::perf::OpenGLGetDeviceClockState();
    nv::perf::OpenGLSetDeviceClockState(NVPW_DEVICE_CLOCK_SETTING_LOCK_TO_RATED_TDP);
    nvperf.StartCollectionOnNextFrame();
#endif

    return true;
}


void megamol::moldyn_gl::rendering::SRTest::release() {
#ifdef USE_NVPERF
    nvperf.Reset();
    nv::perf::OpenGLSetDeviceClockState(clockStatus);
#endif

    glDeleteBuffers(1, &ubo_);
}


std::shared_ptr<glowl::FramebufferObject> create_fbo(std::shared_ptr<glowl::FramebufferObject> const& org_fbo) {
    auto fbo = std::make_shared<glowl::FramebufferObject>(
        org_fbo->getWidth(), org_fbo->getHeight(), glowl::FramebufferObject::DEPTH24_STENCIL8);
    fbo->createColorAttachment(GL_RGB32F, GL_RGB, GL_UNSIGNED_BYTE);
    return fbo;
}


void blit_fbo(std::shared_ptr<glowl::FramebufferObject>& org, std::shared_ptr<glowl::FramebufferObject>& dest) {
    org->bindToRead(0);
    dest->bindToDraw();
    glBlitFramebuffer(0, 0, org->getWidth(), org->getHeight(), 0, 0, dest->getWidth(), dest->getHeight(),
        GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}


bool megamol::moldyn_gl::rendering::SRTest::Render(megamol::mmstd_gl::CallRender3DGL& cr) {
#ifdef MEGAMOL_USE_PROFILING
    auto& pm = const_cast<frontend_resources::PerformanceManager&>(
        frontend_resources.get<frontend_resources::PerformanceManager>());
#endif

#ifdef USE_NVPERF
    nvperf.OnFrameStart();
#endif

    // Camera
    core::view::Camera cam = cr.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cr_fbo = cr.GetFramebuffer();
    //auto cr_fbo = create_fbo(cr_fbo_org);

    // Lights
    glm::vec3 curlightDir = glm::vec3(0.0f, 0.0f, 1.0f);

    auto call_light = getLightsSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        if (distant_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[SphereRenderer] Only one single 'Distant Light' source is supported by this renderer");
        } else if (distant_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[SphereRenderer] No 'Distant Light' found");
        }

        for (auto const& light : distant_lights) {
            auto use_eyedir = light.eye_direction;
            if (use_eyedir) {
                curlightDir = cam_pose.direction;
            } else {
                auto lightDir = light.direction;
                /*core::utility::log::Log::DefaultLog.WriteInfo(
                    "[SRTest] Light dir: (%f, %f, %f)", lightDir[0], lightDir[1], lightDir[2]);*/
                if (lightDir.size() == 3) {
                    curlightDir[0] = lightDir[0];
                    curlightDir[1] = lightDir[1];
                    curlightDir[2] = lightDir[2];
                }
                if (lightDir.size() == 4) {
                    curlightDir[3] = lightDir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                // this->curlightDir = this->curMVtransp * this->curlightDir;
            }
            /// TODO Implement missing distant light parameters:
            // light.second.dl_angularDiameter;
            // light.second.lightColor;
            // light.second.lightIntensity;
        }
    }

    // data
    auto in_call = data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (in_call == nullptr)
        return false;
    in_call->SetFrameID(cr.Time());
    if (!(*in_call)(0))
        return false;

    auto const use_con_ras = use_con_ras_slot_.Param<core::param::BoolParam>()->Value();

    bool new_data = false;
    if (in_data_hash_ != in_call->DataHash() || frame_id_ != in_call->FrameID() || upload_mode_slot_.IsDirty() /*||
        method_slot_.IsDirty()*/) {
        //rt->cleanup();
        loadData(*in_call);
        update_upload_setting();
        in_data_hash_ = in_call->DataHash();
        frame_id_ = in_call->FrameID();
        new_data = true;
        upload_mode_slot_.ResetDirty();
        method_slot_.ResetDirty();
    }
    auto method = static_cast<method_e>(method_slot_.Param<core::param::EnumParam>()->Value());
    auto& rt = rendering_tasks_[method];

    if (method_slot_.IsDirty()) {
        new_data = true;
        method_slot_.ResetDirty();
    }


    if (!(old_cam_ == cam)) {
        ubo_params_t ubo_st;
        ubo_st.dir = cam_pose.direction;
        ubo_st.up = cam_pose.up;
        // ubo_st.right = cam_pose.right;
        ubo_st.right = glm::normalize(glm::cross(cam_pose.direction, cam_pose.up));
        ubo_st.pos = cam_pose.position;
        auto mvp = proj * view;
        ubo_st.mvp = mvp;
        ubo_st.mvp_inv = glm::inverse(mvp);
        ubo_st.attr = glm::vec4(0.f, 0.f, cr_fbo->getWidth(), cr_fbo->getHeight());
        ubo_st.light_dir = curlightDir;
        ubo_st.near_ = cam.get<core::view::Camera::NearPlane>();
        ubo_st.far_ = cam.get<core::view::Camera::FarPlane>();
        ubo_st.p2_z = proj[2].z;
        ubo_st.p3_z = proj[3].z;

        auto const y_angle = static_cast<float>(cam.get<core::view::Camera::FieldOfViewY>());
        auto const ratio = static_cast<float>(cam.get<core::view::Camera::AspectRatio>());
        auto const x_angle = y_angle * ratio;

        ubo_st.frustum_ratio_x = 1.0f / std::cos(x_angle);
        ubo_st.frustum_ratio_y = 1.0f / std::cos(y_angle);
        ubo_st.frustum_ratio_w = std::tan(x_angle);
        ubo_st.frustum_ratio_h = std::tan(y_angle);

        glNamedBufferSubData(ubo_, 0, sizeof(ubo_params_t), &ubo_st);

        old_cam_ = cam;
    }

#ifdef MEGAMOL_USE_CUDA
#if 1
    if (method == method_e::THRUST) {
        SphereRasterizer sr;
        SphereRasterizer::config_t sr_cfg;
        sr_cfg.res = glm::uvec2(cr_fbo->getWidth(), cr_fbo->getHeight());
        sr_cfg.fres = glm::vec2(cr_fbo->getWidth(), cr_fbo->getHeight());
        sr_cfg.global_radius = 0.5f;
        sr_cfg.MVP = proj * view;
        sr_cfg.MVPinv = glm::inverse(sr_cfg.MVP);
        sr_cfg.near_far = glm::vec2(cam.get<core::view::Camera::NearPlane>(), cam.get<core::view::Camera::FarPlane>());
        sr_cfg.camDir = cam_pose.direction;
        sr_cfg.camPos = cam_pose.position;
        sr_cfg.camRight = glm::normalize(glm::cross(cam_pose.direction, cam_pose.up));
        sr_cfg.camUp = cam_pose.up;
        sr_cfg.fovy = cam.get<core::view::Camera::FieldOfViewY>();
        sr_cfg.ratio = cam.get<core::view::Camera::AspectRatio>();
        sr_cfg.lightDir = curlightDir;
        sr_cfg.lower = glm::vec3(in_call->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetX(),
            in_call->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetY(),
            in_call->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetZ());
        sr_cfg.upper = glm::vec3(in_call->AccessBoundingBoxes().ObjectSpaceBBox().GetRightTopFront().GetX(),
            in_call->AccessBoundingBoxes().ObjectSpaceBBox().GetRightTopFront().GetY(),
            in_call->AccessBoundingBoxes().ObjectSpaceBBox().GetRightTopFront().GetZ());
        sr.pm = &pm;
        sr.timing_handles_ = &timing_handles_;

        std::vector<float> radii;
        auto const pl_count = in_call->GetParticleListCount();
        for (int i = 0; i < pl_count; ++i) {
            auto const rad = in_call->AccessParticles(i).GetGlobalRadius();
            radii.push_back(rad);
        }

        auto img_data = sr.Compute(sr_cfg, data_, radii);
        stbi_write_png("sr_img.png", cr_fbo->getWidth(), cr_fbo->getHeight(), 4, img_data.data(),
            cr_fbo->getWidth() * sizeof(glm::u8vec4));
        //std::exit(0);

        in_call->Unlock();

        return true;
    }
#endif
#endif

    // data_.pl_data.clip_distance = clip_thres_slot_.Param<core::param::FloatParam>()->Value();
#ifdef USE_NVPERF
    nvperf.PushRange("Draw_Full");
#endif
    cr_fbo->bind();
    
    if (new_data || enforce_upload_slot_.Param<core::param::BoolParam>()->Value() /* || clip_thres_slot_.IsDirty()*/) {
#ifdef MEGAMOL_USE_PROFILING
        pm.set_transient_comment(
            timing_handles_[0], method_strings[static_cast<method_ut>(method)] + std::string(" ") +
                                    upload_mode_string[static_cast<upload_mode_ut>(rt->get_mode())]);
        pm.start_timer(timing_handles_[0]);
#endif
        //#ifdef USE_NVPERF
        //        nvperf.PushRange("Upload");
        //#endif
        if (method == method_e::COPY) {
            std::dynamic_pointer_cast<copy_rt>(rt)->upload(data_, pm, timing_handles_[2]);
        } else if (method == method_e::COPY_VERT) {
            std::dynamic_pointer_cast<copy_vert_rt>(rt)->upload(data_, pm, timing_handles_[2]);
        } else {
            rt->upload(data_);
        }

//#ifdef USE_NVPERF
//        nvperf.PopRange();
//#endif
#ifdef MEGAMOL_USE_PROFILING
        pm.stop_timer(timing_handles_[0]);
#endif

        new_data = false;
        // clip_thres_slot_.ResetDirty();
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
#ifdef __SRTEST_CON_RAS__
    if (use_con_ras) {
        glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
    }
#endif
#ifdef MEGAMOL_USE_PROFILING
    pm.set_transient_comment(timing_handles_[1], method_strings[static_cast<method_ut>(method)] + std::string(" ") +
                                                     upload_mode_string[static_cast<upload_mode_ut>(rt->get_mode())]);
    pm.start_timer(timing_handles_[1]);
#endif

#ifdef USE_NVPERF
    nvperf.PushRange((method_strings[static_cast<method_ut>(method)] + std::string(" ") +
                      upload_mode_string[static_cast<upload_mode_ut>(rt->get_mode())])
                         .c_str());
#endif

    rt->render(ubo_);

#ifdef USE_NVPERF
    nvperf.PopRange();
#endif

#ifdef MEGAMOL_USE_PROFILING
    pm.stop_timer(timing_handles_[1]);
#endif
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDisable(GL_PROGRAM_POINT_SIZE);
#ifdef __SRTEST_CON_RAS__
    if (use_con_ras) {
        glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
    }
#endif

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

#ifdef USE_NVPERF
    nvperf.PopRange();
#endif

#ifdef USE_NVPERF
    if (!nvperf.IsCollectingReport())
        std::exit(0);

    nvperf.OnFrameEnd();
#endif

    in_call->Unlock();

    return true;
}


bool megamol::moldyn_gl::rendering::SRTest::GetExtents(megamol::mmstd_gl::CallRender3DGL& call) {
    auto cr = &call;
    if (cr == nullptr)
        return false;

    auto c2 = this->data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if ((c2 != nullptr)) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(1))
            return false;
        cr->SetTimeFramesCount(c2->FrameCount());
        auto const plcount = c2->GetParticleListCount();

        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


void megamol::moldyn_gl::rendering::SRTest::loadData(geocalls::MultiParticleDataCall& in_data) {
    core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] Loading Data");


    lower_ = glm::vec3(std::numeric_limits<float>::max());
    upper_ = glm::vec3(std::numeric_limits<float>::lowest());

    auto const pl_count = in_data.GetParticleListCount();

    data_.positions.resize(pl_count);
    data_.colors.resize(pl_count);

    data_.x.resize(pl_count);
    data_.y.resize(pl_count);
    data_.z.resize(pl_count);
    data_.rad.resize(pl_count);
    data_.r.resize(pl_count);
    data_.g.resize(pl_count);
    data_.b.resize(pl_count);
    data_.a.resize(pl_count);
    data_.indices.resize(pl_count);

    data_.data_sizes.resize(pl_count);
    data_.pl_data.global_radii.resize(pl_count);
    data_.pl_data.global_color.resize(pl_count);
    data_.pl_data.use_global_radii.resize(pl_count);
    data_.pl_data.use_global_color.resize(pl_count);
    // data_.pl_data.clip_distance = clip_thres_slot_.Param<core::param::FloatParam>()->Value();

    auto mode = static_cast<upload_mode>(upload_mode_slot_.Param<core::param::EnumParam>()->Value());

    data_.bufArray.resize(pl_count);

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_data.AccessParticles(pl_idx);
        auto& positions = data_.positions[pl_idx];
        auto& colors = data_.colors[pl_idx];

        auto& X = data_.x[pl_idx];
        auto& Y = data_.y[pl_idx];
        auto& Z = data_.z[pl_idx];
        auto& RAD = data_.rad[pl_idx];
        auto& R = data_.r[pl_idx];
        auto& G = data_.g[pl_idx];
        auto& B = data_.b[pl_idx];
        auto& A = data_.a[pl_idx];

        auto& IDX = data_.indices[pl_idx];

        if (parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_NONE &&
            parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I &&
            parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I) {
            data_.pl_data.use_global_color[pl_idx] = 0;
        } else {
            data_.pl_data.use_global_color[pl_idx] = 1;
            data_.pl_data.global_color[pl_idx] = glm::vec4(1,
                0, 0, 1);
        }

        if (parts.GetVertexDataType() != geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
            data_.pl_data.use_global_radii[pl_idx] = 1;
            data_.pl_data.global_radii[pl_idx] = parts.GetGlobalRadius();
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[SRTest] Having global radius: %f", data_.pl_data.global_radii[pl_idx]);
        } else {
            data_.pl_data.use_global_radii[pl_idx] = 0;
            core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] Having no global radius");
        }

        auto const p_count = parts.GetCount();
        positions.clear();
        positions.reserve(p_count * 4);
        colors.clear();
        colors.reserve(p_count * 4);

        #define ADDED_STUFF

#ifdef ADDED_STUFF
        X.clear();
        X.reserve(p_count);
        Y.clear();
        Y.reserve(p_count);
        Z.clear();
        Z.reserve(p_count);
        RAD.clear();
        RAD.reserve(p_count);
        R.clear();
        R.reserve(p_count);
        G.clear();
        G.reserve(p_count);
        B.clear();
        B.reserve(p_count);
        A.clear();
        A.reserve(p_count);
        IDX.clear();
        IDX.reserve(p_count * 6);
#endif

        data_.data_sizes[pl_idx] = p_count;

        auto const& x_acc = parts.GetParticleStore().GetXAcc();
        auto const& y_acc = parts.GetParticleStore().GetYAcc();
        auto const& z_acc = parts.GetParticleStore().GetZAcc();
        auto const& rad_acc = parts.GetParticleStore().GetRAcc();
        auto const& cr_acc = parts.GetParticleStore().GetCRAcc();
        auto const& cg_acc = parts.GetParticleStore().GetCGAcc();
        auto const& cb_acc = parts.GetParticleStore().GetCBAcc();
        auto const& ca_acc = parts.GetParticleStore().GetCAAcc();

        std::array<uint8_t, 4> default_col = {1, 0, 0, 1};

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto pos = glm::vec3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));
            auto rad = rad_acc->Get_f(p_idx);

            lower_ = glm::min(lower_, pos);
            upper_ = glm::max(upper_, pos);

            positions.push_back(pos.x);
            positions.push_back(pos.y);
            positions.push_back(pos.z);
            if (mode == upload_mode::POS_COL_SEP || mode == upload_mode::FULL_SEP) {
                positions.push_back(rad);
                colors.push_back(cr_acc->Get_f(p_idx));
                colors.push_back(cg_acc->Get_f(p_idx));
                colors.push_back(cb_acc->Get_f(p_idx));
                colors.push_back(ca_acc->Get_f(p_idx));
            } else if (mode == upload_mode::VEC3_SEP) {
                unsigned int col = glm::packUnorm4x8(
                    glm::vec4(cr_acc->Get_f(p_idx), cg_acc->Get_f(p_idx), cb_acc->Get_f(p_idx), ca_acc->Get_f(p_idx)));
                colors.push_back(*reinterpret_cast<float*>(&col));
            } else if (mode == upload_mode::NO_SEP || mode == upload_mode::BUFFER_ARRAY) {
                unsigned int col =
                    glm::packUnorm4x8(glm::vec4(default_col[0], default_col[1], default_col[2], default_col[3]));
                positions.push_back(*reinterpret_cast<float*>(&col));
            }

#ifdef ADDED_STUFF
            X.push_back(pos.x);
            Y.push_back(pos.y);
            Z.push_back(pos.z);
            RAD.push_back(rad);
            R.push_back(cr_acc->Get_f(p_idx));
            G.push_back(cg_acc->Get_f(p_idx));
            B.push_back(cb_acc->Get_f(p_idx));
            A.push_back(ca_acc->Get_f(p_idx));

            IDX.push_back(p_idx * 4 + 0);
            IDX.push_back(p_idx * 4 + 1);
            IDX.push_back(p_idx * 4 + 2);
            IDX.push_back(p_idx * 4 + 3);
            IDX.push_back(p_idx * 4 + 3);
            IDX.push_back(p_idx * 4 + 4);
#endif
        }

        if (mode == upload_mode::BUFFER_ARRAY) {
            auto& bufA = data_.bufArray[pl_idx];
            bufA.SetDataWithSize(positions.data(), 16, 16, parts.GetCount(), (GLuint)(2 * 1024 * 1024 * 1024 - 1));
        }
    }
}


megamol::moldyn_gl::rendering::vao_rt::vao_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task(upload_mode::NULL_MODE, "SRTestVAO", options, std::filesystem::path("srtest/srtest.vert.glsl"),
              std::filesystem::path("srtest/srtest.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::vao_rt::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vao = vaos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, num_prims);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::vao_rt::upload(data_package_t const& package) {
    auto const num_vaos = package.positions.size();
    glDeleteVertexArrays(vaos_.size(), vaos_.data());
    vaos_.resize(num_vaos);
    glCreateVertexArrays(vaos_.size(), vaos_.data());

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_vaos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_vaos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_vaos)> i = 0; i < num_vaos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);
        glEnableVertexArrayAttrib(vaos_[i], 0);
        glVertexArrayAttribFormat(vaos_[i], 0, 4, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayVertexBuffer(vaos_[i], 0, vbos_[0], 0, 4 * sizeof(float));
        glVertexArrayAttribBinding(vaos_[i], 0, 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
        glEnableVertexArrayAttrib(vaos_[i], 1);
        glVertexArrayAttribFormat(vaos_[i], 1, 4, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayVertexBuffer(vaos_[i], 1, cbos_[0], 0, 4 * sizeof(float));
        glVertexArrayAttribBinding(vaos_[i], 1, 1);
    }

    pl_data_ = package.pl_data;

    return true;
}


bool megamol::moldyn_gl::rendering::vao_rt::cleanup() {
    if (!vaos_.empty())
        glDeleteVertexArrays(vaos_.size(), vaos_.data());

    if (!vbos_.empty())
        glDeleteBuffers(vbos_.size(), vbos_.data());

    if (!cbos_.empty())
        glDeleteBuffers(cbos_.size(), cbos_.data());

    return true;
}


megamol::moldyn_gl::rendering::ssbo_rt::ssbo_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_points, "SRTestSSBO", options, std::filesystem::path("srtest/srtest.vert.glsl"),
              std::filesystem::path("srtest/srtest.frag.glsl")) {}


megamol::moldyn_gl::rendering::ssbo_geo_rt::ssbo_geo_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_points, "SRTestSSBOGeo", options,
              std::filesystem::path("srtest/srtest_geo.vert.glsl"),
              std::filesystem::path("srtest/srtest_geo.geom.glsl"),
              std::filesystem::path("srtest/srtest_geo.frag.glsl")) {}


megamol::moldyn_gl::rendering::ssbo_vert_rt::ssbo_vert_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_verts, "SRTestSSBOVert", options,
              std::filesystem::path("srtest/srtest_vert.vert.glsl"),
              std::filesystem::path("srtest/srtest_vert.frag.glsl")) {}


megamol::moldyn_gl::rendering::mesh_rt::mesh_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_mesh, "SRTestMesh", options, std::filesystem::path("srtest/srtest_mesh.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh.frag.glsl")) {}


megamol::moldyn_gl::rendering::mesh_geo_altn_rt::mesh_geo_altn_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_mesh, "SRTestMeshGeoAltn", options,
              std::filesystem::path("srtest/srtest_mesh_geo_altn.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_geo_altn.frag.glsl")) {}


megamol::moldyn_gl::rendering::mesh_geo_rt::mesh_geo_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_mesh, "SRTestMeshGeo", options,
              std::filesystem::path("srtest/srtest_mesh_geo.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_geo.frag.glsl")) {}


megamol::moldyn_gl::rendering::mesh_altn_rt::mesh_altn_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_mesh, "SRTestMeshAltn", options,
              std::filesystem::path("srtest/srtest_mesh_altn.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_altn.frag.glsl")) {}


megamol::moldyn_gl::rendering::mesh_geo_task_rt::mesh_geo_task_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : mesh_shader_task("SRTestMeshGeoTask", options, std::filesystem::path("srtest/srtest_mesh_geo_task.task.glsl"),
              std::filesystem::path("srtest/srtest_mesh_geo_task.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_geo.frag.glsl")) {}


megamol::moldyn_gl::rendering::tex_rt::tex_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task(upload_mode::NULL_MODE, "SRTestTex", options, std::filesystem::path("srtest/srtest.vert.glsl"),
              std::filesystem::path("srtest/srtest.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::tex_rt::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto tex = tex_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);
        // program->setUniform("clip_dist", pl_data_.clip_distance);

        program->setUniform("data_tex", 1);
        program->setUniform("num_points", static_cast<unsigned int>(num_prims));
        auto base_size = sqrt(num_prims);
        program->setUniform("base_size", static_cast<unsigned int>(base_size));

        glBindBuffer(GL_TEXTURE_BUFFER, buf_[i]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER, tex_[i]);
        glDrawArrays(GL_POINTS, 0, num_prims);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::tex_rt::upload(data_package_t const& package) {
    int value;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
    uint32_t num_tex = package.positions.size();

    glDeleteTextures(tex_.size(), tex_.data());
    tex_.resize(num_tex);
    glGenTextures(tex_.size(), tex_.data());

    glDeleteBuffers(buf_.size(), buf_.data());
    buf_.resize(num_tex);
    glCreateBuffers(buf_.size(), buf_.data());

    num_prims_ = package.data_sizes;
    glActiveTexture(GL_TEXTURE0);
    for (uint32_t i = 0; i < package.positions.size(); ++i) {
        glNamedBufferStorage(buf_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        auto total_tex_size = package.positions[i].size() / 4;
        auto base_size = std::floor(sqrt(total_tex_size));
        auto left_size = total_tex_size / base_size;
        glBindTexture(GL_TEXTURE_BUFFER, tex_[i]);
        // glTextureStorage2D(tex_[i], 0, GL_RGBA32F, base_size, left_size);

        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, buf_[i]);

        // glTextureBuffer(tex_[i], GL_R32F, buf_[i]);
        /*glTextureSubImage2D(tex_[i], 0, 0, 0, base_size, left_size, GL_RGBA, GL_FLOAT,
            package.positions[i].data());*/
        glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // glBindTextureUnit(0, tex_[i]);
    }
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    pl_data_ = package.pl_data;

    return true;
}

bool megamol::moldyn_gl::rendering::tex_rt::cleanup() {
    if (!tex_.empty())
        glDeleteTextures(tex_.size(), tex_.data());

    if (!buf_.empty())
        glDeleteBuffers(buf_.size(), buf_.data());

    return true;
}


megamol::moldyn_gl::rendering::copy_rt::copy_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task(upload_mode::NULL_MODE, "SRTestCopy", options,
              std::filesystem::path("srtest/srtest.vert.glsl"), std::filesystem::path("srtest/srtest.frag.glsl")) {
    try {
        comp_program_ = core::utility::make_glowl_shader(
            "SRTestCopyComp", options, std::filesystem::path("srtest/srtest_copy.comp.glsl"));
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteError("[copy_rt] Failed to create program");
        throw;
    }
}


megamol::moldyn_gl::rendering::copy_rt::~copy_rt() {
    cleanup();
}


bool megamol::moldyn_gl::rendering::copy_rt::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);
    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, copy_bos_[i]);

        glDrawArrays(GL_POINTS, 0, num_prims);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::copy_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(copy_bos_.size(), copy_bos_.data());
    copy_bos_.resize(num_ssbos);
    glCreateBuffers(copy_bos_.size(), copy_bos_.data());

    glDeleteBuffers(xbos_.size(), xbos_.data());
    xbos_.resize(num_ssbos);
    glCreateBuffers(xbos_.size(), xbos_.data());

    glDeleteBuffers(ybos_.size(), ybos_.data());
    ybos_.resize(num_ssbos);
    glCreateBuffers(ybos_.size(), ybos_.data());

    glDeleteBuffers(zbos_.size(), zbos_.data());
    zbos_.resize(num_ssbos);
    glCreateBuffers(zbos_.size(), zbos_.data());

    glDeleteBuffers(radbos_.size(), radbos_.data());
    radbos_.resize(num_ssbos);
    glCreateBuffers(radbos_.size(), radbos_.data());

    glDeleteBuffers(rbos_.size(), rbos_.data());
    rbos_.resize(num_ssbos);
    glCreateBuffers(rbos_.size(), rbos_.data());

    glDeleteBuffers(gbos_.size(), gbos_.data());
    gbos_.resize(num_ssbos);
    glCreateBuffers(gbos_.size(), gbos_.data());

    glDeleteBuffers(bbos_.size(), bbos_.data());
    bbos_.resize(num_ssbos);
    glCreateBuffers(bbos_.size(), bbos_.data());

    glDeleteBuffers(abos_.size(), abos_.data());
    abos_.resize(num_ssbos);
    glCreateBuffers(abos_.size(), abos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferData(copy_bos_[i], package.x[i].size() * 16, nullptr, GL_STATIC_COPY);

        glNamedBufferStorage(xbos_[i], package.x[i].size() * sizeof(std::decay_t<decltype(package.x[i])>::value_type),
            package.x[i].data(), 0);

        glNamedBufferStorage(ybos_[i], package.y[i].size() * sizeof(std::decay_t<decltype(package.y[i])>::value_type),
            package.y[i].data(), 0);

        glNamedBufferStorage(zbos_[i], package.z[i].size() * sizeof(std::decay_t<decltype(package.z[i])>::value_type),
            package.z[i].data(), 0);

        glNamedBufferStorage(radbos_[i],
            package.rad[i].size() * sizeof(std::decay_t<decltype(package.rad[i])>::value_type), package.rad[i].data(),
            0);

        glNamedBufferStorage(rbos_[i], package.r[i].size() * sizeof(std::decay_t<decltype(package.r[i])>::value_type),
            package.r[i].data(), 0);

        glNamedBufferStorage(gbos_[i], package.g[i].size() * sizeof(std::decay_t<decltype(package.g[i])>::value_type),
            package.g[i].data(), 0);

        glNamedBufferStorage(bbos_[i], package.b[i].size() * sizeof(std::decay_t<decltype(package.b[i])>::value_type),
            package.b[i].data(), 0);

        glNamedBufferStorage(abos_[i], package.a[i].size() * sizeof(std::decay_t<decltype(package.a[i])>::value_type),
            package.a[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    comp_program_->use();
    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];
        comp_program_->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ybos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, zbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, radbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, gbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, abos_[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, copy_bos_[i]);

        glDispatchCompute(num_prims / 32 + 1, 1, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    return true;
}

bool megamol::moldyn_gl::rendering::copy_rt::upload(data_package_t const& package,
    frontend_resources::PerformanceManager& pm, frontend_resources::PerformanceManager::handle_type handle) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(copy_bos_.size(), copy_bos_.data());
    copy_bos_.resize(num_ssbos);
    glCreateBuffers(copy_bos_.size(), copy_bos_.data());

    glDeleteBuffers(xbos_.size(), xbos_.data());
    xbos_.resize(num_ssbos);
    glCreateBuffers(xbos_.size(), xbos_.data());

    glDeleteBuffers(ybos_.size(), ybos_.data());
    ybos_.resize(num_ssbos);
    glCreateBuffers(ybos_.size(), ybos_.data());

    glDeleteBuffers(zbos_.size(), zbos_.data());
    zbos_.resize(num_ssbos);
    glCreateBuffers(zbos_.size(), zbos_.data());

    glDeleteBuffers(radbos_.size(), radbos_.data());
    radbos_.resize(num_ssbos);
    glCreateBuffers(radbos_.size(), radbos_.data());

    glDeleteBuffers(rbos_.size(), rbos_.data());
    rbos_.resize(num_ssbos);
    glCreateBuffers(rbos_.size(), rbos_.data());

    glDeleteBuffers(gbos_.size(), gbos_.data());
    gbos_.resize(num_ssbos);
    glCreateBuffers(gbos_.size(), gbos_.data());

    glDeleteBuffers(bbos_.size(), bbos_.data());
    bbos_.resize(num_ssbos);
    glCreateBuffers(bbos_.size(), bbos_.data());

    glDeleteBuffers(abos_.size(), abos_.data());
    abos_.resize(num_ssbos);
    glCreateBuffers(abos_.size(), abos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferData(copy_bos_[i], package.x[i].size() * 16, nullptr, GL_STATIC_COPY);

        glNamedBufferStorage(xbos_[i], package.x[i].size() * sizeof(std::decay_t<decltype(package.x[i])>::value_type),
            package.x[i].data(), 0);

        glNamedBufferStorage(ybos_[i], package.y[i].size() * sizeof(std::decay_t<decltype(package.y[i])>::value_type),
            package.y[i].data(), 0);

        glNamedBufferStorage(zbos_[i], package.z[i].size() * sizeof(std::decay_t<decltype(package.z[i])>::value_type),
            package.z[i].data(), 0);

        glNamedBufferStorage(radbos_[i],
            package.rad[i].size() * sizeof(std::decay_t<decltype(package.rad[i])>::value_type), package.rad[i].data(),
            0);

        glNamedBufferStorage(rbos_[i], package.r[i].size() * sizeof(std::decay_t<decltype(package.r[i])>::value_type),
            package.r[i].data(), 0);

        glNamedBufferStorage(gbos_[i], package.g[i].size() * sizeof(std::decay_t<decltype(package.g[i])>::value_type),
            package.g[i].data(), 0);

        glNamedBufferStorage(bbos_[i], package.b[i].size() * sizeof(std::decay_t<decltype(package.b[i])>::value_type),
            package.b[i].data(), 0);

        glNamedBufferStorage(abos_[i], package.a[i].size() * sizeof(std::decay_t<decltype(package.a[i])>::value_type),
            package.a[i].data(), 0);
    }

    pl_data_ = package.pl_data;

#ifdef MEGAMOL_USE_PROFILING
    pm.set_transient_comment(handle, "COPY");
    pm.start_timer(handle);
#endif

    comp_program_->use();
    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];
        comp_program_->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ybos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, zbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, radbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, gbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, abos_[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, copy_bos_[i]);

        glDispatchCompute(num_prims / 32 + 1, 1, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

#ifdef MEGAMOL_USE_PROFILING
    pm.stop_timer(handle);
#endif

    return true;
}

bool megamol::moldyn_gl::rendering::copy_rt::cleanup() {
    if (!copy_bos_.empty())
        glDeleteBuffers(copy_bos_.size(), copy_bos_.data());

    if (!xbos_.empty())
        glDeleteBuffers(xbos_.size(), xbos_.data());

    if (!ybos_.empty())
        glDeleteBuffers(ybos_.size(), ybos_.data());

    if (!zbos_.empty())
        glDeleteBuffers(zbos_.size(), zbos_.data());

    if (!radbos_.empty())
        glDeleteBuffers(radbos_.size(), radbos_.data());

    if (!rbos_.empty())
        glDeleteBuffers(rbos_.size(), rbos_.data());

    if (!gbos_.empty())
        glDeleteBuffers(gbos_.size(), gbos_.data());

    if (!bbos_.empty())
        glDeleteBuffers(bbos_.size(), bbos_.data());

    if (!abos_.empty())
        glDeleteBuffers(abos_.size(), abos_.data());

    return true;
}


megamol::moldyn_gl::rendering::copy_vert_rt::copy_vert_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task(upload_mode::NULL_MODE, "SRTestCopyVert", options,
              std::filesystem::path("srtest/srtest_vert.vert.glsl"),
              std::filesystem::path("srtest/srtest_vert.frag.glsl")) {
    try {
        comp_program_ = core::utility::make_glowl_shader(
            "SRTestCopyVertComp", options, std::filesystem::path("srtest/srtest_copy.comp.glsl"));
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteError("[copy_rt] Failed to create program");
        throw;
    }
}


megamol::moldyn_gl::rendering::copy_vert_rt::~copy_vert_rt() {
    cleanup();
}


bool megamol::moldyn_gl::rendering::copy_vert_rt::render(GLuint ubo) {
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);
    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, copy_bos_[i]);

        glDrawArrays(GL_TRIANGLES, 0, num_prims * 6);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);

    return true;
}


bool megamol::moldyn_gl::rendering::copy_vert_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(copy_bos_.size(), copy_bos_.data());
    copy_bos_.resize(num_ssbos);
    glCreateBuffers(copy_bos_.size(), copy_bos_.data());

    glDeleteBuffers(xbos_.size(), xbos_.data());
    xbos_.resize(num_ssbos);
    glCreateBuffers(xbos_.size(), xbos_.data());

    glDeleteBuffers(ybos_.size(), ybos_.data());
    ybos_.resize(num_ssbos);
    glCreateBuffers(ybos_.size(), ybos_.data());

    glDeleteBuffers(zbos_.size(), zbos_.data());
    zbos_.resize(num_ssbos);
    glCreateBuffers(zbos_.size(), zbos_.data());

    glDeleteBuffers(radbos_.size(), radbos_.data());
    radbos_.resize(num_ssbos);
    glCreateBuffers(radbos_.size(), radbos_.data());

    glDeleteBuffers(rbos_.size(), rbos_.data());
    rbos_.resize(num_ssbos);
    glCreateBuffers(rbos_.size(), rbos_.data());

    glDeleteBuffers(gbos_.size(), gbos_.data());
    gbos_.resize(num_ssbos);
    glCreateBuffers(gbos_.size(), gbos_.data());

    glDeleteBuffers(bbos_.size(), bbos_.data());
    bbos_.resize(num_ssbos);
    glCreateBuffers(bbos_.size(), bbos_.data());

    glDeleteBuffers(abos_.size(), abos_.data());
    abos_.resize(num_ssbos);
    glCreateBuffers(abos_.size(), abos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferData(copy_bos_[i], package.x[i].size() * 16, nullptr, GL_STATIC_COPY);

        glNamedBufferStorage(xbos_[i], package.x[i].size() * sizeof(std::decay_t<decltype(package.x[i])>::value_type),
            package.x[i].data(), 0);

        glNamedBufferStorage(ybos_[i], package.y[i].size() * sizeof(std::decay_t<decltype(package.y[i])>::value_type),
            package.y[i].data(), 0);

        glNamedBufferStorage(zbos_[i], package.z[i].size() * sizeof(std::decay_t<decltype(package.z[i])>::value_type),
            package.z[i].data(), 0);

        glNamedBufferStorage(radbos_[i],
            package.rad[i].size() * sizeof(std::decay_t<decltype(package.rad[i])>::value_type), package.rad[i].data(),
            0);

        glNamedBufferStorage(rbos_[i], package.r[i].size() * sizeof(std::decay_t<decltype(package.r[i])>::value_type),
            package.r[i].data(), 0);

        glNamedBufferStorage(gbos_[i], package.g[i].size() * sizeof(std::decay_t<decltype(package.g[i])>::value_type),
            package.g[i].data(), 0);

        glNamedBufferStorage(bbos_[i], package.b[i].size() * sizeof(std::decay_t<decltype(package.b[i])>::value_type),
            package.b[i].data(), 0);

        glNamedBufferStorage(abos_[i], package.a[i].size() * sizeof(std::decay_t<decltype(package.a[i])>::value_type),
            package.a[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    comp_program_->use();
    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];
        comp_program_->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ybos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, zbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, radbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, gbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, abos_[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, copy_bos_[i]);

        glDispatchCompute(num_prims / 32 + 1, 1, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    return true;
}

bool megamol::moldyn_gl::rendering::copy_vert_rt::upload(data_package_t const& package,
    frontend_resources::PerformanceManager& pm, frontend_resources::PerformanceManager::handle_type handle) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(copy_bos_.size(), copy_bos_.data());
    copy_bos_.resize(num_ssbos);
    glCreateBuffers(copy_bos_.size(), copy_bos_.data());

    glDeleteBuffers(xbos_.size(), xbos_.data());
    xbos_.resize(num_ssbos);
    glCreateBuffers(xbos_.size(), xbos_.data());

    glDeleteBuffers(ybos_.size(), ybos_.data());
    ybos_.resize(num_ssbos);
    glCreateBuffers(ybos_.size(), ybos_.data());

    glDeleteBuffers(zbos_.size(), zbos_.data());
    zbos_.resize(num_ssbos);
    glCreateBuffers(zbos_.size(), zbos_.data());

    glDeleteBuffers(radbos_.size(), radbos_.data());
    radbos_.resize(num_ssbos);
    glCreateBuffers(radbos_.size(), radbos_.data());

    glDeleteBuffers(rbos_.size(), rbos_.data());
    rbos_.resize(num_ssbos);
    glCreateBuffers(rbos_.size(), rbos_.data());

    glDeleteBuffers(gbos_.size(), gbos_.data());
    gbos_.resize(num_ssbos);
    glCreateBuffers(gbos_.size(), gbos_.data());

    glDeleteBuffers(bbos_.size(), bbos_.data());
    bbos_.resize(num_ssbos);
    glCreateBuffers(bbos_.size(), bbos_.data());

    glDeleteBuffers(abos_.size(), abos_.data());
    abos_.resize(num_ssbos);
    glCreateBuffers(abos_.size(), abos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferData(copy_bos_[i], package.x[i].size() * 16, nullptr, GL_STATIC_COPY);

        glNamedBufferStorage(xbos_[i], package.x[i].size() * sizeof(std::decay_t<decltype(package.x[i])>::value_type),
            package.x[i].data(), 0);

        glNamedBufferStorage(ybos_[i], package.y[i].size() * sizeof(std::decay_t<decltype(package.y[i])>::value_type),
            package.y[i].data(), 0);

        glNamedBufferStorage(zbos_[i], package.z[i].size() * sizeof(std::decay_t<decltype(package.z[i])>::value_type),
            package.z[i].data(), 0);

        glNamedBufferStorage(radbos_[i],
            package.rad[i].size() * sizeof(std::decay_t<decltype(package.rad[i])>::value_type), package.rad[i].data(),
            0);

        glNamedBufferStorage(rbos_[i], package.r[i].size() * sizeof(std::decay_t<decltype(package.r[i])>::value_type),
            package.r[i].data(), 0);

        glNamedBufferStorage(gbos_[i], package.g[i].size() * sizeof(std::decay_t<decltype(package.g[i])>::value_type),
            package.g[i].data(), 0);

        glNamedBufferStorage(bbos_[i], package.b[i].size() * sizeof(std::decay_t<decltype(package.b[i])>::value_type),
            package.b[i].data(), 0);

        glNamedBufferStorage(abos_[i], package.a[i].size() * sizeof(std::decay_t<decltype(package.a[i])>::value_type),
            package.a[i].data(), 0);
    }

    pl_data_ = package.pl_data;

#ifdef MEGAMOL_USE_PROFILING
    pm.set_transient_comment(handle, "COPY_VERT");
    pm.start_timer(handle);
#endif

    comp_program_->use();
    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];
        comp_program_->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ybos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, zbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, radbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, gbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bbos_[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, abos_[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, copy_bos_[i]);

        glDispatchCompute(num_prims / 32 + 1, 1, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

#ifdef MEGAMOL_USE_PROFILING
    pm.stop_timer(handle);
#endif

    return true;
}


bool megamol::moldyn_gl::rendering::copy_vert_rt::cleanup() {
    if (!copy_bos_.empty())
        glDeleteBuffers(copy_bos_.size(), copy_bos_.data());

    if (!xbos_.empty())
        glDeleteBuffers(xbos_.size(), xbos_.data());

    if (!ybos_.empty())
        glDeleteBuffers(ybos_.size(), ybos_.data());

    if (!zbos_.empty())
        glDeleteBuffers(zbos_.size(), zbos_.data());

    if (!radbos_.empty())
        glDeleteBuffers(radbos_.size(), radbos_.data());

    if (!rbos_.empty())
        glDeleteBuffers(rbos_.size(), rbos_.data());

    if (!gbos_.empty())
        glDeleteBuffers(gbos_.size(), gbos_.data());

    if (!bbos_.empty())
        glDeleteBuffers(bbos_.size(), bbos_.data());

    if (!abos_.empty())
        glDeleteBuffers(abos_.size(), abos_.data());

    return true;
}


megamol::moldyn_gl::rendering::ssbo_quad_rt::ssbo_quad_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_quads, "SRTestSSBOQuad", options,
              std::filesystem::path("srtest/srtest_vert.vert.glsl"),
              std::filesystem::path("srtest/srtest_vert.frag.glsl")) {}


megamol::moldyn_gl::rendering::ssbo_strip_rt::ssbo_strip_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_strip, "SRTestSSBOStrip", options,
              std::filesystem::path("srtest/srtest_vert.vert.glsl"),
              std::filesystem::path("srtest/srtest_vert.frag.glsl")) {}


megamol::moldyn_gl::rendering::ssbo_muzic_rt::ssbo_muzic_rt(
    upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options)
        : ssbo_shader_task(mode, dc_strip, "SRTestSSBOMuzic", options,
              std::filesystem::path("srtest/srtest_vert.vert.glsl"),
              std::filesystem::path("srtest/srtest_vert.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::ssbo_muzic_rt::render(GLuint ubo) {
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        switch (get_mode()) {
        case upload_mode::FULL_SEP: {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ybos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, zbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, radbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, gbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, abos_[i]);
        } break;
        case upload_mode::NO_SEP: {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbos_[i]);
        } break;
        case upload_mode::POS_COL_SEP:
        case upload_mode::VEC3_SEP:
        default:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbos_[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbos_[i]);
        }

        // glDrawArrays(GL_POINTS, 0, num_prims);
        //dc_muzic(num_prims, indices_[i]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ind_buf_[i]);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cmd_buf_[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, offset_buf_[i]);

        glMultiDrawElementsIndirect(GL_TRIANGLE_STRIP, GL_UNSIGNED_INT, 0, cmd_count_[i], 0);

        /*constexpr int per_iter = 10000000;
        auto num_iter = num_prims / per_iter + 1;
        for (int iter = 0; iter < num_iter; ++iter) {
            auto num_items = iter * per_iter;
            num_items = std::fmin(num_prims - num_items, per_iter);
            program->setUniform("offset", iter * per_iter);
            glDrawElements(GL_TRIANGLE_STRIP, num_items * 6 - 2, GL_UNSIGNED_INT, nullptr);
        }*/
    }
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    return true;
}


bool megamol::moldyn_gl::rendering::ssbo_muzic_rt::upload(data_package_t const& package) {
    ssbo_shader_task::upload(package);

    indices_ = package.indices;

    typedef struct {
        GLuint count;
        GLuint instanceCount;
        GLuint firstIndex;
        GLuint baseVertex;
        GLuint baseInstance;
    } DrawElementsIndirectCommand;

    glDeleteBuffers(ind_buf_.size(), ind_buf_.data());
    ind_buf_.resize(indices_.size());
    glCreateBuffers(ind_buf_.size(), ind_buf_.data());

    glDeleteBuffers(cmd_buf_.size(), cmd_buf_.data());
    cmd_buf_.resize(indices_.size());
    glCreateBuffers(cmd_buf_.size(), cmd_buf_.data());

    glDeleteBuffers(offset_buf_.size(), offset_buf_.data());
    offset_buf_.resize(indices_.size());
    glCreateBuffers(offset_buf_.size(), offset_buf_.data());

    cmd_count_.resize(indices_.size());

    for (int i = 0; i < indices_.size(); ++i) {
        glNamedBufferStorage(ind_buf_[i], sizeof(unsigned int) * indices_[i].size(), indices_[i].data(), 0);

        std::vector<uint32_t> offsets;
        std::vector<DrawElementsIndirectCommand> commands;

        constexpr int per_iter = 1000;
        auto num_iter = num_prims_[i] / per_iter;
        offsets.reserve(num_iter);
        commands.reserve(num_iter);
        cmd_count_[i] = num_iter;
        for (int iter = 0; iter < num_iter; ++iter) {
            auto num_items = iter * per_iter;
            num_items = std::fmin(num_prims_[i] - num_items, per_iter);
            offsets.push_back(iter * per_iter);
            DrawElementsIndirectCommand cmd;
            memset(&cmd, 0, sizeof(DrawElementsIndirectCommand));
            cmd.count = num_items * 6 - 2;
            cmd.instanceCount = 1;
            commands.push_back(cmd);
            //program->setUniform("offset", iter * per_iter);
            //glDrawElements(GL_TRIANGLE_STRIP, num_items * 6 - 2, GL_UNSIGNED_INT, nullptr);
        }
        if (num_iter * per_iter < num_prims_[i]) {
            offsets.push_back(num_iter * per_iter);
            DrawElementsIndirectCommand cmd;
            memset(&cmd, 0, sizeof(DrawElementsIndirectCommand));
            cmd.count = (num_prims_[i] - num_iter * per_iter) * 6 - 2;
            cmd.instanceCount = 1;
            commands.push_back(cmd);
        }

        glNamedBufferStorage(cmd_buf_[i], sizeof(DrawElementsIndirectCommand) * commands.size(), commands.data(), 0);
        glNamedBufferStorage(offset_buf_[i], sizeof(uint32_t) * offsets.size(), offsets.data(), 0);
    }

    return true;
}


bool megamol::moldyn_gl::rendering::ssbo_muzic_rt::cleanup() {
    if (!ind_buf_.empty())
        glDeleteBuffers(ind_buf_.size(), ind_buf_.data());

    if (!cmd_buf_.empty())
        glDeleteBuffers(cmd_buf_.size(), cmd_buf_.data());

    if (!offset_buf_.empty())
        glDeleteBuffers(offset_buf_.size(), offset_buf_.data());

    return true;
}
