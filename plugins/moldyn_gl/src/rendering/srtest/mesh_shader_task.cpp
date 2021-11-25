#include "mesh_shader_task.h"


bool megamol::moldyn_gl::rendering::mesh_shader_task::render(GLuint ubo) {
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


bool megamol::moldyn_gl::rendering::mesh_shader_task::upload(data_package_t const& package) {
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
