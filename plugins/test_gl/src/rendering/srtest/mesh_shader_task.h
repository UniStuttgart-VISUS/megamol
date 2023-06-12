/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "rendering_task.h"

namespace megamol::test_gl::rendering {
class mesh_shader_task : public rendering_task {
public:
    template<typename... Paths>
    mesh_shader_task(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths)
            : rendering_task(upload_mode::NULL_MODE, label, options, std::forward<Paths>(paths)...) {}

    virtual ~mesh_shader_task();

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

    bool cleanup() override;

private:
    std::vector<GLuint> vbos_;
    std::vector<GLuint> cbos_;

    std::vector<GLuint> xbos_;
    std::vector<GLuint> ybos_;
    std::vector<GLuint> zbos_;
    std::vector<GLuint> radbos_;
    std::vector<GLuint> rbos_;
    std::vector<GLuint> gbos_;
    std::vector<GLuint> bbos_;
    std::vector<GLuint> abos_;

    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;

    std::vector<core::utility::SSBOBufferArray> const* bufArray;
};
} // namespace megamol::test_gl::rendering
