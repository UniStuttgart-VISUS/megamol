#pragma once

#include "rendering_task.h"

namespace megamol::moldyn_gl::rendering {
class mesh_shader_task : public rendering_task {
public:
    template<typename... Paths>
    mesh_shader_task(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths)
            : rendering_task(label, options, std::forward<Paths>(paths)...) {}

    virtual ~mesh_shader_task() = default;

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

private:
    std::vector<GLuint> vbos_;
    std::vector<GLuint> cbos_;
    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;
};
} // namespace megamol::moldyn_gl::rendering
