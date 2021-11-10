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

    return true;
}


void megamol::moldyn_gl::rendering::SRTest::release() {}


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

    param_package_t param;
    param.front = cam_pose.direction;
    param.up = cam_pose.up;
    param.right = cam_pose.right;
    param.pos = cam_pose.position;
    param.rad = 0.5f;
    param.global_col = glm::vec4(0.f, 1.f, 0.f, 1.f);
    auto mvp = proj * view;
    param.mvp = mvp;
    param.proj = proj;
    param.mvp_inv = glm::inverse(mvp);
    param.mv_inv = glm::inverse(view);
    param.mvp_trans = glm::transpose(mvp);
    param.attr = glm::vec4(0.f, 0.f, cr_fbo->getWidth(), cr_fbo->getHeight());
    param.light_dir = glm::vec3(0.f, 0.f, 1.f);
    param.near_ = cam.get<core::view::Camera::NearPlane>();
    param.far_ = cam.get<core::view::Camera::FarPlane>();

    cr_fbo->bind();

    GLuint64 startTime, midTime, stopTime;
    GLuint queryID[3];
    glGenQueries(3, queryID);

    glQueryCounter(queryID[0], GL_TIMESTAMP);

    rt->upload(data_);

    glQueryCounter(queryID[1], GL_TIMESTAMP);

    rt->render(param);

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

    data_.data.resize(pl_count);
    data_.data_sizes.resize(pl_count);

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_data.AccessParticles(pl_idx);
        auto& data = data_.data[pl_idx];

        auto const p_count = parts.GetCount();
        data.clear();
        data.reserve(p_count * 3);
        data_.data_sizes[pl_idx] = p_count;

        auto const& x_acc = parts.GetParticleStore().GetXAcc();
        auto const& y_acc = parts.GetParticleStore().GetYAcc();
        auto const& z_acc = parts.GetParticleStore().GetZAcc();
        auto const& rad_acc = parts.GetParticleStore().GetRAcc();

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            data.push_back(x_acc->Get_f(p_idx));
            data.push_back(y_acc->Get_f(p_idx));
            data.push_back(z_acc->Get_f(p_idx));
            data.push_back(rad_acc->Get_f(p_idx));
        }
    }
}


megamol::moldyn_gl::rendering::vao_rt::vao_rt(msf::ShaderFactoryOptionsOpenGL const& options)
        : rendering_task("SRTestVAO", options, std::filesystem::path("srtest/vao.vert.glsl"),
              std::filesystem::path("srtest/vao.frag.glsl")) {}


bool megamol::moldyn_gl::rendering::vao_rt::render(param_package_t const& package) {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    auto program = get_program();
    program->use();

    program->setUniform("viewAttr", package.attr);
    program->setUniform("lightDir", package.light_dir);
    program->setUniform("camIn", package.front);
    program->setUniform("camUp", glm::normalize(package.up));
    program->setUniform("camPos", package.pos);
    program->setUniform("camRight", package.right);
    program->setUniform("constRad", package.rad);
    program->setUniform("MVP", package.mvp);
    program->setUniform("P", package.proj);
    program->setUniform("MVPinv", package.mvp_inv);
    program->setUniform("MVinv", package.mv_inv);
    program->setUniform("MVPtransp", package.mvp_trans);
    program->setUniform("globalCol", package.global_col);
    program->setUniform("near", package.near_);
    program->setUniform("far", package.far_);


    for (int i = 0; i < num_prims_.size(); ++i) {
        auto vao = vaos_[i];
        auto num_prims = num_prims_[i];
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, num_prims);
    }
    glBindVertexArray(0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::vao_rt::upload(data_package_t const& package) {
    auto const num_vaos = package.data.size();
    glDeleteVertexArrays(vaos_.size(), vaos_.data());
    vaos_.resize(num_vaos);
    glGenVertexArrays(vaos_.size(), vaos_.data());

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_vaos);
    glGenBuffers(vbos_.size(), vbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_vaos)> i = 0; i < num_vaos; ++i) {
        glBindVertexArray(vaos_[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbos_[i]);
        glBufferData(GL_ARRAY_BUFFER,
            package.data[i].size() * sizeof(std::decay_t<decltype(package.data[i])>::value_type),
            package.data[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return true;
}
