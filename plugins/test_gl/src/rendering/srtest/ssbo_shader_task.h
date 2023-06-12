/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>

#include "rendering_task.h"

namespace megamol::test_gl::rendering {

using draw_cmd_t = std::function<void(unsigned int)>;

class ssbo_shader_task : public rendering_task {
public:
    template<typename... Paths>
    ssbo_shader_task(upload_mode const& mode, draw_cmd_t const& draw_cmd, std::string const& label,
        msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths)
            : rendering_task(mode, label, options, std::forward<Paths>(paths)...)
            , draw_cmd_(draw_cmd) {}

    virtual ~ssbo_shader_task();

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

    bool cleanup() override;

protected:
    void upload_full_separate(data_package_t const& package);

    void upload_pos_col_sep(data_package_t const& package);

    void upload_no_sep(data_package_t const& package);

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

    draw_cmd_t draw_cmd_;

    std::vector<core::utility::SSBOBufferArray> const* bufArray;
};

} // namespace megamol::test_gl::rendering
