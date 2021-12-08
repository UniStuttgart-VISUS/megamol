#include "SRTest.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore/view/light/DistantLight.h"


megamol::moldyn_gl::rendering::SRTest::SRTest()
        : data_in_slot_("inData", "")
        , getLightsSlot("lights", "Lights are retrieved over this slot.")
        , method_slot_("method", "")
        , upload_mode_slot_("upload mode", "")
        , enforce_upload_slot_("enforce upload", "")
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
    method_slot_ << ep;
    MakeSlotAvailable(&method_slot_);

    ep = new core::param::EnumParam(static_cast<upload_mode_ut>(upload_mode::FULL_SEP));
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::FULL_SEP), "FULL_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::POS_COL_SEP), "POS_COL_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::VEC3_SEP), "VEC3_SEP");
    ep->SetTypePair(static_cast<upload_mode_ut>(upload_mode::NO_SEP), "NO_SEP");
    upload_mode_slot_ << ep;
    MakeSlotAvailable(&upload_mode_slot_);

    enforce_upload_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&enforce_upload_slot_);

    /*clip_thres_slot_ << new core::param::FloatParam(0.00001f, 0.0f);
    MakeSlotAvailable(&clip_thres_slot_);*/
}


megamol::moldyn_gl::rendering::SRTest::~SRTest() {
    this->Release();
}


bool megamol::moldyn_gl::rendering::SRTest::create_shaders() {
    try {
        auto shdr_vao_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_vao_options.addDefinition("__SRTEST_VAO__");
        auto shdr_tex_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_tex_options.addDefinition("__SRTEST_TEX__");
        auto shdr_ssbo_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_ssbo_options.addDefinition("__SRTEST_SSBO__");
        auto shdr_ssbo_vert_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_ssbo_vert_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_vert_options.addDefinition("BASE_IDX", VERT_BASE_IDX);
        shdr_ssbo_vert_options.addDefinition("INV_IDX", VERT_INV_IDX);
        shdr_ssbo_vert_options.addDefinition("BUMP_IDX", VERT_BUMP_IDX);
        auto shdr_ssbo_quads_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_ssbo_quads_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_quads_options.addDefinition("BASE_IDX", QUADS_BASE_IDX);
        shdr_ssbo_quads_options.addDefinition("INV_IDX", QUADS_INV_IDX);
        shdr_ssbo_quads_options.addDefinition("BUMP_IDX", QUADS_BUMP_IDX);
        auto shdr_ssbo_strip_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_ssbo_strip_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_strip_options.addDefinition("BASE_IDX", STRIP_BASE_IDX);
        shdr_ssbo_strip_options.addDefinition("INV_IDX", STRIP_INV_IDX);
        shdr_ssbo_strip_options.addDefinition("BUMP_IDX", STRIP_BUMP_IDX);
        auto shdr_ssbo_muzic_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_ssbo_muzic_options.addDefinition("__SRTEST_SSBO__");
        shdr_ssbo_muzic_options.addDefinition("BASE_IDX", MUZIC_BASE_IDX);
        shdr_ssbo_muzic_options.addDefinition("INV_IDX", MUZIC_INV_IDX);
        shdr_ssbo_muzic_options.addDefinition("BUMP_IDX", MUZIC_BUMP_IDX);
        auto shdr_copy_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_copy_options.addDefinition("__SRTEST_SSBO__");
        shdr_copy_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
        auto shdr_copy_vert_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_copy_vert_options.addDefinition("__SRTEST_SSBO__");
        shdr_copy_vert_options.addDefinition("__SRTEST_UPLOAD_NO_SEP__");
        shdr_copy_vert_options.addDefinition("BASE_IDX", VERT_BASE_IDX);
        shdr_copy_vert_options.addDefinition("INV_IDX", VERT_INV_IDX);
        shdr_copy_vert_options.addDefinition("BUMP_IDX", VERT_BUMP_IDX);
        auto shdr_mesh_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_options.addDefinition("__SRTEST_MESH__");
        shdr_mesh_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto shdr_mesh_altn_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_altn_options.addDefinition("__SRTEST_MESH_ALTN__");
        shdr_mesh_altn_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto shdr_mesh_geo_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_geo_options.addDefinition("__SRTEST_MESH_GEO__");
        shdr_mesh_geo_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto shdr_mesh_geo_altn_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_geo_altn_options.addDefinition("__SRTEST_MESH_GEO_ALTN__");
        shdr_mesh_geo_altn_options.addDefinition("WARP", std::to_string(MESH_WARP_SIZE));
        auto mode = static_cast<upload_mode>(upload_mode_slot_.Param<core::param::EnumParam>()->Value());
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
        }
        }

        rendering_tasks_.insert(std::make_pair(method_e::VAO, std::make_unique<vao_rt>(shdr_vao_options)));

        rendering_tasks_.insert(std::make_pair(method_e::TEX, std::make_unique<tex_rt>(shdr_tex_options)));

        rendering_tasks_.insert(std::make_pair(method_e::COPY, std::make_unique<copy_rt>(shdr_copy_options)));

        rendering_tasks_.insert(
            std::make_pair(method_e::COPY_VERT, std::make_unique<copy_vert_rt>(shdr_copy_vert_options)));

        rendering_tasks_.insert(std::make_pair(method_e::SSBO, std::make_unique<ssbo_rt>(mode, shdr_ssbo_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_GEO, std::make_unique<ssbo_geo_rt>(mode, shdr_ssbo_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_VERT, std::make_unique<ssbo_vert_rt>(mode, shdr_ssbo_vert_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_QUAD, std::make_unique<ssbo_quad_rt>(mode, shdr_ssbo_quads_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_STRIP, std::make_unique<ssbo_strip_rt>(mode, shdr_ssbo_strip_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::SSBO_MUZIC, std::make_unique<ssbo_muzic_rt>(mode, shdr_ssbo_muzic_options)));

        rendering_tasks_.insert(std::make_pair(method_e::MESH, std::make_unique<mesh_rt>(mode, shdr_mesh_options)));

        rendering_tasks_.insert(
            std::make_pair(method_e::MESH_ALTN, std::make_unique<mesh_altn_rt>(mode, shdr_mesh_altn_options)));

        rendering_tasks_.insert(
            std::make_pair(method_e::MESH_GEO, std::make_unique<mesh_geo_rt>(mode, shdr_mesh_geo_options)));
        rendering_tasks_.insert(
            std::make_pair(method_e::MESH_GEO_TASK, std::make_unique<mesh_geo_task_rt>(mode, shdr_mesh_geo_options)));

        rendering_tasks_.insert(std::make_pair(
            method_e::MESH_GEO_ALTN, std::make_unique<mesh_geo_altn_rt>(mode, shdr_mesh_geo_altn_options)));
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
#ifdef PROFILING
    auto& pm = const_cast<frontend_resources::PerformanceManager&>(
        frontend_resources.get<frontend_resources::PerformanceManager>());
    frontend_resources::PerformanceManager::basic_timer_config upload_timer, render_timer;
    upload_timer.name = "upload";
    upload_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timing_handles_ = pm.add_timers(this, {upload_timer, render_timer});
#endif
    if (!create_shaders())
        return false;

    glCreateBuffers(1, &ubo_);
    glNamedBufferData(ubo_, sizeof(ubo_params_t), nullptr, GL_DYNAMIC_DRAW);

    GLint max_vert;
    GLint max_ind;
    glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &max_vert);
    glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &max_ind);

    core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] Max Vert %d; Max Ind %d", max_vert, max_ind);

    return true;
}


void megamol::moldyn_gl::rendering::SRTest::release() {
    glDeleteBuffers(1, &ubo_);
}


bool megamol::moldyn_gl::rendering::SRTest::Render(megamol::core_gl::view::CallRender3DGL& cr) {
#ifdef PROFILING
    auto& pm = const_cast<frontend_resources::PerformanceManager&>(
        frontend_resources.get<frontend_resources::PerformanceManager>());
#endif

    // Camera
    core::view::Camera cam = cr.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cr_fbo = cr.GetFramebuffer();

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

    bool new_data = false;
    if (in_data_hash_ != in_call->DataHash() || frame_id_ != in_call->FrameID() || upload_mode_slot_.IsDirty()) {
        loadData(*in_call);
        update_upload_setting();
        in_data_hash_ = in_call->DataHash();
        frame_id_ = in_call->FrameID();
        new_data = true;
        upload_mode_slot_.ResetDirty();
    }

    auto method = static_cast<method_e>(method_slot_.Param<core::param::EnumParam>()->Value());
    if (method_slot_.IsDirty()) {
        new_data = true;
        method_slot_.ResetDirty();
    }

    /*if (upload_mode_slot_.IsDirty()) {
        update_upload_setting();
        new_data = true;
        upload_mode_slot_.ResetDirty();
    }*/


    auto& rt = rendering_tasks_[method];


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

        /*core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] x_angle %f y_angle %f ratio %f",
            static_cast<float>(x_angle), static_cast<float>(y_angle), static_cast<float>(ratio));*/

        ubo_st.frustum_ratio_x = 1.0f / std::cosf(x_angle);
        ubo_st.frustum_ratio_y = 1.0f / std::cosf(y_angle);
        ubo_st.frustum_ratio_w = std::tanf(x_angle);
        ubo_st.frustum_ratio_h = std::tanf(y_angle);

        glNamedBufferSubData(ubo_, 0, sizeof(ubo_params_t), &ubo_st);

        old_cam_ = cam;
    }

    // data_.pl_data.clip_distance = clip_thres_slot_.Param<core::param::FloatParam>()->Value();

    cr_fbo->bind();

    /*GLuint64 startTime, midTime, stopTime;
    GLuint queryID[3];
    glGenQueries(3, queryID);

    glQueryCounter(queryID[0], GL_TIMESTAMP);*/
    if (new_data || enforce_upload_slot_.Param<core::param::BoolParam>()->Value() /* || clip_thres_slot_.IsDirty()*/) {
#ifdef PROFILING
        pm.set_transient_comment(
            timing_handles_[0], method_strings[static_cast<method_ut>(method)] + std::string(" ") +
                                    upload_mode_string[static_cast<upload_mode_ut>(rt->get_mode())]);
        pm.start_timer(timing_handles_[0], this->GetCoreInstance()->GetFrameID());
#endif

        rt->upload(data_);

#ifdef PROFILING
        pm.stop_timer(timing_handles_[0]);
#endif

        new_data = false;
        // clip_thres_slot_.ResetDirty();
    }

#ifdef PROFILING
    // glQueryCounter(queryID[1], GL_TIMESTAMP);
    pm.set_transient_comment(timing_handles_[1], method_strings[static_cast<method_ut>(method)] + std::string(" ") +
                                                     upload_mode_string[static_cast<upload_mode_ut>(rt->get_mode())]);
    pm.start_timer(timing_handles_[1], this->GetCoreInstance()->GetFrameID());
#endif

    rt->render(ubo_);

#ifdef PROFILING
    pm.stop_timer(timing_handles_[1]);
#endif

    // glQueryCounter(queryID[2], GL_TIMESTAMP);

    /*GLint query_complete = false;
    while (!query_complete) {
        glGetQueryObjectiv(queryID[2], GL_QUERY_RESULT_AVAILABLE, &query_complete);
    }
    glGetQueryObjectui64v(queryID[0], GL_QUERY_RESULT, &startTime);
    glGetQueryObjectui64v(queryID[1], GL_QUERY_RESULT, &midTime);
    glGetQueryObjectui64v(queryID[2], GL_QUERY_RESULT, &stopTime);

    glDeleteQueries(3, queryID);*/

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    /*core::utility::log::Log::DefaultLog.WriteInfo(
        "[SRTest] Upload time: %d Render time: %d", midTime - startTime, stopTime - midTime);*/

    return true;
}


bool megamol::moldyn_gl::rendering::SRTest::GetExtents(megamol::core_gl::view::CallRender3DGL& call) {
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
    // this->curClipBox = cr->AccessBoundingBoxes().ClipBox();

    return true;
}


void megamol::moldyn_gl::rendering::SRTest::loadData(geocalls::MultiParticleDataCall& in_data) {
    core::utility::log::Log::DefaultLog.WriteInfo("[SRTest] Loading Data");

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

        if (parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_NONE) {
            data_.pl_data.use_global_color[pl_idx] = 0;
        } else {
            data_.pl_data.use_global_color[pl_idx] = 1;
            data_.pl_data.global_color[pl_idx] = glm::vec4(parts.GetGlobalColour()[0], parts.GetGlobalColour()[1],
                parts.GetGlobalColour()[2], parts.GetGlobalColour()[3]);
        }

        if (parts.GetVertexDataType() != geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
            data_.pl_data.use_global_radii[pl_idx] = 1;
            data_.pl_data.global_radii[pl_idx] = parts.GetGlobalRadius();
        } else {
            data_.pl_data.use_global_radii[pl_idx] = 0;
        }

        auto const p_count = parts.GetCount();
        positions.clear();
        positions.reserve(p_count * 4);
        colors.clear();
        colors.reserve(p_count * 4);

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

        data_.data_sizes[pl_idx] = p_count;

        auto const& x_acc = parts.GetParticleStore().GetXAcc();
        auto const& y_acc = parts.GetParticleStore().GetYAcc();
        auto const& z_acc = parts.GetParticleStore().GetZAcc();
        auto const& rad_acc = parts.GetParticleStore().GetRAcc();
        auto const& cr_acc = parts.GetParticleStore().GetCRAcc();
        auto const& cg_acc = parts.GetParticleStore().GetCGAcc();
        auto const& cb_acc = parts.GetParticleStore().GetCBAcc();
        auto const& ca_acc = parts.GetParticleStore().GetCAAcc();

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            positions.push_back(x_acc->Get_f(p_idx));
            positions.push_back(y_acc->Get_f(p_idx));
            positions.push_back(z_acc->Get_f(p_idx));
            if (mode == upload_mode::POS_COL_SEP || mode == upload_mode::FULL_SEP) {
                positions.push_back(rad_acc->Get_f(p_idx));
                colors.push_back(cr_acc->Get_f(p_idx));
                colors.push_back(cg_acc->Get_f(p_idx));
                colors.push_back(cb_acc->Get_f(p_idx));
                colors.push_back(ca_acc->Get_f(p_idx));
            } else if (mode == upload_mode::VEC3_SEP) {
                unsigned int col = glm::packUnorm4x8(
                    glm::vec4(cr_acc->Get_f(p_idx), cg_acc->Get_f(p_idx), cb_acc->Get_f(p_idx), ca_acc->Get_f(p_idx)));
                colors.push_back(*reinterpret_cast<float*>(&col));
            } else if (mode == upload_mode::NO_SEP) {
                unsigned int col = glm::packUnorm4x8(
                    glm::vec4(cr_acc->Get_f(p_idx), cg_acc->Get_f(p_idx), cb_acc->Get_f(p_idx), ca_acc->Get_f(p_idx)));
                positions.push_back(*reinterpret_cast<float*>(&col));
            }

            X.push_back(x_acc->Get_f(p_idx));
            Y.push_back(y_acc->Get_f(p_idx));
            Z.push_back(z_acc->Get_f(p_idx));
            RAD.push_back(rad_acc->Get_f(p_idx));
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
        auto base_size = std::floorf(sqrt(total_tex_size));
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
    glDeleteBuffers(copy_bos_.size(), copy_bos_.data());

    glDeleteBuffers(xbos_.size(), xbos_.data());

    glDeleteBuffers(ybos_.size(), ybos_.data());

    glDeleteBuffers(zbos_.size(), zbos_.data());

    glDeleteBuffers(radbos_.size(), radbos_.data());

    glDeleteBuffers(rbos_.size(), rbos_.data());

    glDeleteBuffers(gbos_.size(), gbos_.data());

    glDeleteBuffers(bbos_.size(), bbos_.data());

    glDeleteBuffers(abos_.size(), abos_.data());
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
    glDeleteBuffers(copy_bos_.size(), copy_bos_.data());

    glDeleteBuffers(xbos_.size(), xbos_.data());

    glDeleteBuffers(ybos_.size(), ybos_.data());

    glDeleteBuffers(zbos_.size(), zbos_.data());

    glDeleteBuffers(radbos_.size(), radbos_.data());

    glDeleteBuffers(rbos_.size(), rbos_.data());

    glDeleteBuffers(gbos_.size(), gbos_.data());

    glDeleteBuffers(bbos_.size(), bbos_.data());

    glDeleteBuffers(abos_.size(), abos_.data());
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

        glDrawArrays(GL_QUADS, 0, num_prims * 4);
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

        constexpr int per_iter = 100000;
        auto num_iter = num_prims / per_iter + 1;
        for (int iter = 0; iter < num_iter; ++iter) {
            auto num_items = iter * per_iter;
            num_items = std::fmin(num_prims - num_items, per_iter);
            program->setUniform("offset", iter * per_iter);
            glDrawElements(GL_TRIANGLE_STRIP, num_items * 6 - 2, GL_UNSIGNED_INT, nullptr);
        }
    }
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

    glDeleteBuffers(ind_buf_.size(), ind_buf_.data());
    ind_buf_.resize(indices_.size());
    glCreateBuffers(ind_buf_.size(), ind_buf_.data());

    for (int i = 0; i < indices_.size(); ++i) {
        glNamedBufferStorage(ind_buf_[i], sizeof(unsigned int) * indices_[i].size(), indices_[i].data(), 0);
    }

    return true;
}
