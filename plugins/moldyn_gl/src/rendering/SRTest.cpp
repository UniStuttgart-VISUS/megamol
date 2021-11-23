#include "SRTest.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore/view/light/DistantLight.h"


megamol::moldyn_gl::rendering::SRTest::SRTest()
        : data_in_slot_("inData", "")
        , getLightsSlot("lights", "Lights are retrieved over this slot.")
        , method_slot_("method", "")
        , clip_thres_slot_("clip distance", "") {
    data_in_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightsSlot);

    auto ep = new core::param::EnumParam(static_cast<method_ut>(method_e::VAO));
    ep->SetTypePair(static_cast<method_ut>(method_e::VAO), "VAO");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO), "SSBO");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_GEO), "SSBO_GEO");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO_VERT), "SSBO_VERT");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH), "MESH");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_ALTN), "MESH_ALTN");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_GEO), "MESH_GEO");
    ep->SetTypePair(static_cast<method_ut>(method_e::MESH_GEO_ALTN), "MESH_GEO_ALTN");
    method_slot_ << ep;
    MakeSlotAvailable(&method_slot_);

    clip_thres_slot_ << new core::param::FloatParam(0.00001f, 0.0f);
    MakeSlotAvailable(&clip_thres_slot_);
}


megamol::moldyn_gl::rendering::SRTest::~SRTest() {
    this->Release();
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
    try {
        auto shdr_vao_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_vao_options.addDefinition("__SRTEST_VAO__");
        rendering_tasks_.insert(std::make_pair(method_e::VAO, std::make_unique<vao_rt>(shdr_vao_options)));
        auto shdr_ssbo_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_ssbo_options.addDefinition("__SRTEST_SSBO__");
        rendering_tasks_.insert(std::make_pair(method_e::SSBO, std::make_unique<ssbo_rt>(shdr_ssbo_options)));
        rendering_tasks_.insert(std::make_pair(method_e::SSBO_GEO, std::make_unique<ssbo_geo_rt>(shdr_ssbo_options)));
        rendering_tasks_.insert(std::make_pair(method_e::SSBO_VERT, std::make_unique<ssbo_vert_rt>(shdr_ssbo_options)));
        auto shdr_mesh_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_options.addDefinition("__SRTEST_MESH__");
        rendering_tasks_.insert(std::make_pair(method_e::MESH, std::make_unique<mesh_rt>(shdr_mesh_options)));
        auto shdr_mesh_altn_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_altn_options.addDefinition("__SRTEST_MESH_ALTN__");
        rendering_tasks_.insert(
            std::make_pair(method_e::MESH_ALTN, std::make_unique<mesh_altn_rt>(shdr_mesh_altn_options)));
        auto shdr_mesh_geo_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_geo_options.addDefinition("__SRTEST_MESH_GEO__");
        rendering_tasks_.insert(
            std::make_pair(method_e::MESH_GEO, std::make_unique<mesh_geo_rt>(shdr_mesh_geo_options)));
        auto shdr_mesh_geo_altn_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        shdr_mesh_geo_altn_options.addDefinition("__SRTEST_MESH_GEO_ALTN__");
        rendering_tasks_.insert(
            std::make_pair(method_e::MESH_GEO_ALTN, std::make_unique<mesh_geo_altn_rt>(shdr_mesh_geo_altn_options)));
    } catch (glowl::GLSLProgramException const& e) {
        core::utility::log::Log::DefaultLog.WriteError("[SRTest] %s", e.what());
        return false;
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteError("[SRTest] Failed to create shader programs");
        return false;
    }

    glCreateBuffers(1, &ubo_);
    glNamedBufferData(ubo_, sizeof(ubo_params_t), nullptr, GL_DYNAMIC_DRAW);

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
    if (in_data_hash_ != in_call->DataHash() || frame_id_ != in_call->FrameID()) {
        loadData(*in_call);
        in_data_hash_ = in_call->DataHash();
        frame_id_ = in_call->FrameID();
        new_data = true;
    }

    auto method = static_cast<method_e>(method_slot_.Param<core::param::EnumParam>()->Value());
    if (method_slot_.IsDirty()) {
        new_data = true;
        method_slot_.ResetDirty();
    }

    auto& rt = rendering_tasks_[method];


    if (!(old_cam_ == cam)) {
        ubo_params_t ubo_st;
        ubo_st.dir = cam_pose.direction;
        ubo_st.up = cam_pose.up;
        ubo_st.right = cam_pose.right;
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
    if (new_data || clip_thres_slot_.IsDirty()) {
#ifdef PROFILING
        pm.start_timer(timing_handles_[0], this->GetCoreInstance()->GetFrameID());
#endif

        rt->upload(data_);

#ifdef PROFILING
        pm.stop_timer(timing_handles_[0]);
#endif

        new_data = false;
        clip_thres_slot_.ResetDirty();
    }

#ifdef PROFILING
    // glQueryCounter(queryID[1], GL_TIMESTAMP);
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
    auto const pl_count = in_data.GetParticleListCount();

    data_.positions.resize(pl_count);
    data_.colors.resize(pl_count);
    data_.data_sizes.resize(pl_count);
    data_.pl_data.global_radii.resize(pl_count);
    data_.pl_data.global_color.resize(pl_count);
    data_.pl_data.use_global_radii.resize(pl_count);
    data_.pl_data.use_global_color.resize(pl_count);
    // data_.pl_data.clip_distance = clip_thres_slot_.Param<core::param::FloatParam>()->Value();

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_data.AccessParticles(pl_idx);
        auto& positions = data_.positions[pl_idx];
        auto& colors = data_.colors[pl_idx];

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
            positions.push_back(rad_acc->Get_f(p_idx));
            colors.push_back(cr_acc->Get_f(p_idx));
            colors.push_back(cg_acc->Get_f(p_idx));
            colors.push_back(cb_acc->Get_f(p_idx));
            colors.push_back(ca_acc->Get_f(p_idx));
        }
    }
}


megamol::moldyn_gl::rendering::vao_rt::vao_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestVAO", options, std::filesystem::path("srtest/srtest.vert.glsl"),
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


megamol::moldyn_gl::rendering::ssbo_rt::ssbo_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestSSBO", options, std::filesystem::path("srtest/srtest.vert.glsl"),
              std::filesystem::path("srtest/srtest.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::ssbo_rt::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
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


bool megamol::moldyn_gl::rendering::ssbo_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}


megamol::moldyn_gl::rendering::mesh_rt::mesh_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestMesh", options, std::filesystem::path("srtest/srtest_mesh.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::mesh_rt::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
        glDrawMeshTasksNV(0, num_prims / 32 + 1);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::mesh_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}


megamol::moldyn_gl::rendering::mesh_geo_altn_rt::mesh_geo_altn_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestMeshGeoAltn", options, std::filesystem::path("srtest/srtest_mesh_geo_altn.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_geo_altn.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::mesh_geo_altn_rt::render(GLuint ubo) {
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
        glDrawMeshTasksNV(0, num_prims / 32 + 1);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);

    return true;
}


bool megamol::moldyn_gl::rendering::mesh_geo_altn_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}


megamol::moldyn_gl::rendering::mesh_geo_rt::mesh_geo_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestMeshGeo", options, std::filesystem::path("srtest/srtest_mesh_geo.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_geo.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::mesh_geo_rt::render(GLuint ubo) {
    glEnable(GL_CLIP_DISTANCE0);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
        glDrawMeshTasksNV(0, num_prims / 32 + 1);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CLIP_DISTANCE0);

    return true;
}


bool megamol::moldyn_gl::rendering::mesh_geo_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}


megamol::moldyn_gl::rendering::mesh_altn_rt::mesh_altn_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestMeshAltn", options, std::filesystem::path("srtest/srtest_mesh_altn.mesh.glsl"),
              std::filesystem::path("srtest/srtest_mesh_altn.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::mesh_altn_rt::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        program->setUniform("num_points", static_cast<unsigned int>(num_prims));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
        glDrawMeshTasksNV(0, num_prims / 32 + 1);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::mesh_altn_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}


megamol::moldyn_gl::rendering::ssbo_geo_rt::ssbo_geo_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestSSBOGeo", options, std::filesystem::path("srtest/srtest_geo.vert.glsl"),
              std::filesystem::path("srtest/srtest_geo.geom.glsl"),
              std::filesystem::path("srtest/srtest_geo.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::ssbo_geo_rt::render(GLuint ubo) {
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
        glDrawArrays(GL_POINTS, 0, num_prims);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);

    return true;
}


bool megamol::moldyn_gl::rendering::ssbo_geo_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}


megamol::moldyn_gl::rendering::ssbo_vert_rt::ssbo_vert_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestSSBOVert", options, std::filesystem::path("srtest/srtest_vert.vert.glsl"),
              std::filesystem::path("srtest/srtest_vert.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::ssbo_vert_rt::render(GLuint ubo) {
    glEnable(GL_CLIP_DISTANCE0);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);

    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vbo = vbos_[i];
        auto cbo = cbos_[i];
        auto num_prims = num_prims_[i];

        program->setUniform("useGlobalCol", pl_data_.use_global_color[i]);
        program->setUniform("useGlobalRad", pl_data_.use_global_radii[i]);
        program->setUniform("globalCol", pl_data_.global_color[i]);
        program->setUniform("globalRad", pl_data_.global_radii[i]);
        // program->setUniform("clip_dist", pl_data_.clip_distance);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cbo);
        glDrawArrays(GL_QUADS, 0, num_prims * 4);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CLIP_DISTANCE0);

    return true;
}


bool megamol::moldyn_gl::rendering::ssbo_vert_rt::upload(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_ssbos);
    glCreateBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);

        glNamedBufferStorage(cbos_[i],
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), 0);
    }

    pl_data_ = package.pl_data;

    return true;
}
