#include "ssbo_shader_task.h"


megamol::moldyn_gl::rendering::ssbo_shader_task::~ssbo_shader_task() {
    cleanup();
}


bool megamol::moldyn_gl::rendering::ssbo_shader_task::render(GLuint ubo) {
    glEnable(GL_PROGRAM_POINT_SIZE);
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
        draw_cmd_(num_prims);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDisable(GL_PROGRAM_POINT_SIZE);

    return true;
}


bool megamol::moldyn_gl::rendering::ssbo_shader_task::upload(data_package_t const& package) {
    switch (get_mode()) {
    case upload_mode::FULL_SEP:
        upload_full_separate(package);
        break;
    case upload_mode::NO_SEP:
        upload_no_sep(package);
        break;
    case upload_mode::POS_COL_SEP:
    case upload_mode::VEC3_SEP:
    default:
        upload_pos_col_sep(package);
    }

    return true;
}

bool megamol::moldyn_gl::rendering::ssbo_shader_task::cleanup() {
    if (!vbos_.empty())
        glDeleteBuffers(vbos_.size(), vbos_.data());

    if (!cbos_.empty())
        glDeleteBuffers(cbos_.size(), cbos_.data());

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


void megamol::moldyn_gl::rendering::ssbo_shader_task::upload_full_separate(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

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
}


void megamol::moldyn_gl::rendering::ssbo_shader_task::upload_pos_col_sep(data_package_t const& package) {
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
}


void megamol::moldyn_gl::rendering::ssbo_shader_task::upload_no_sep(data_package_t const& package) {
    auto const num_ssbos = package.positions.size();

    glDeleteBuffers(vbos_.size(), vbos_.data());
    vbos_.resize(num_ssbos);
    glCreateBuffers(vbos_.size(), vbos_.data());

    num_prims_ = package.data_sizes;

    for (std::decay_t<decltype(num_ssbos)> i = 0; i < num_ssbos; ++i) {
        glNamedBufferStorage(vbos_[i],
            package.positions[i].size() * sizeof(std::decay_t<decltype(package.positions[i])>::value_type),
            package.positions[i].data(), 0);
    }

    pl_data_ = package.pl_data;
}
