#include "SRTest.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"


megamol::moldyn_gl::rendering::SRTest::SRTest() : data_in_slot_("inData", ""), method_slot_("method", "") {
    data_in_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    auto ep = new core::param::EnumParam(static_cast<method_ut>(method_e::VAO));
    ep->SetTypePair(static_cast<method_ut>(method_e::VAO), "VAO");
    ep->SetTypePair(static_cast<method_ut>(method_e::SSBO), "SSBO");
    method_slot_ << ep;
    MakeSlotAvailable(&method_slot_);
}


megamol::moldyn_gl::rendering::SRTest::~SRTest() {
    this->Release();
}


bool megamol::moldyn_gl::rendering::SRTest::create() {
    try {
        auto const shdr_cp_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        rendering_tasks_.insert(std::make_pair(method_e::VAO, std::make_unique<vao_rt>(shdr_cp_options)));
    } catch (glowl::GLSLProgramException const& e) {
        core::utility::log::Log::DefaultLog.WriteError("[SRTest] %s", e.what());
        return false;
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteError("[SRTest] Failed to create shader programs");
        return false;
    }

    glGenBuffers(1, &ubo_);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo_);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ubo_params_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    return true;
}


void megamol::moldyn_gl::rendering::SRTest::release() {
    glDeleteBuffers(1, &ubo_);
}


bool megamol::moldyn_gl::rendering::SRTest::Render(megamol::core::view::CallRender3DGL& cr) {
    // Camera
    core::view::Camera cam = cr.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cr_fbo = cr.GetFramebuffer();

    // data
    auto in_call = data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (in_call == nullptr)
        return false;
    in_call->SetFrameID(cr.Time());
    if (!(*in_call)(0))
        return false;

    if (in_data_hash_ != in_call->DataHash() || frame_id_ != in_call->FrameID()) {
        loadData(*in_call);
        in_data_hash_ = in_call->DataHash();
        frame_id_ = in_call->FrameID();
    }

    auto method = static_cast<method_e>(method_slot_.Param<core::param::EnumParam>()->Value());

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
        ubo_st.mvp_trans = glm::transpose(mvp);
        ubo_st.attr = glm::vec4(0.f, 0.f, cr_fbo->getWidth(), cr_fbo->getHeight());
        ubo_st.light_dir = glm::vec3(0.f, 0.f, 1.f);
        ubo_st.near_ = cam.get<core::view::Camera::NearPlane>();
        ubo_st.far_ = cam.get<core::view::Camera::FarPlane>();

        glBindBuffer(GL_UNIFORM_BUFFER, ubo_);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(ubo_params_t), &ubo_st);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        old_cam_ = cam;
    }

    cr_fbo->bind();

    GLuint64 startTime, midTime, stopTime;
    GLuint queryID[3];
    glGenQueries(3, queryID);

    glQueryCounter(queryID[0], GL_TIMESTAMP);

    rt->upload(data_);

    glQueryCounter(queryID[1], GL_TIMESTAMP);

    rt->render(ubo_);

    glQueryCounter(queryID[2], GL_TIMESTAMP);

    GLint query_complete = false;
    while (!query_complete) {
        glGetQueryObjectiv(queryID[2], GL_QUERY_RESULT_AVAILABLE, &query_complete);
    }
    glGetQueryObjectui64v(queryID[0], GL_QUERY_RESULT, &startTime);
    glGetQueryObjectui64v(queryID[1], GL_QUERY_RESULT, &midTime);
    glGetQueryObjectui64v(queryID[2], GL_QUERY_RESULT, &stopTime);

    glDeleteQueries(3, queryID);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[SRTest] Upload time: %d Render time: %d", midTime - startTime, stopTime - midTime);

    return true;
}


bool megamol::moldyn_gl::rendering::SRTest::GetExtents(megamol::core::view::CallRender3DGL& call) {
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
        : rendering_task("SRTestVAO", options, std::filesystem::path("srtest/vao.vert.glsl"),
              std::filesystem::path("srtest/vao.frag.glsl")) {}


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
    glGenVertexArrays(vaos_.size(), vaos_.data());

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_vaos);
    glGenBuffers(vbos_.size(), vbos_.data());

    glDeleteBuffers(cbos_.size(), cbos_.data());
    cbos_.resize(num_vaos);
    glGenBuffers(cbos_.size(), cbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_vaos)> i = 0; i < num_vaos; ++i) {
        glBindVertexArray(vaos_[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbos_[i]);
        glBufferData(GL_ARRAY_BUFFER,
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, cbos_[i]);
        glBufferData(GL_ARRAY_BUFFER,
            package.colors[i].size() * sizeof(std::decay_t<decltype(package.colors[i])>::value_type),
            package.colors[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    pl_data_ = package.pl_data;

    return true;
}
