/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>

#include "mmcore_gl/utility/SSBOBufferArray.h"
#include "mmcore_gl/utility/ShaderFactory.h"

namespace megamol::test_gl::rendering {
struct per_list_package {
    std::vector<float> global_radii;
    std::vector<glm::vec4> global_color;
    std::vector<uint8_t> use_global_radii;
    std::vector<uint8_t> use_global_color;
};

using per_list_package_t = per_list_package;

struct data_package {
    std::vector<std::vector<float>> positions;
    std::vector<std::vector<float>> colors;
    std::vector<std::vector<float>> x;
    std::vector<std::vector<float>> y;
    std::vector<std::vector<float>> z;
    std::vector<std::vector<float>> rad;
    std::vector<std::vector<float>> r;
    std::vector<std::vector<float>> g;
    std::vector<std::vector<float>> b;
    std::vector<std::vector<float>> a;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<uint64_t> data_sizes;
    per_list_package pl_data;
    std::vector<core::utility::SSBOBufferArray> bufArray;
};

using data_package_t = data_package;

struct ubo_params {
    alignas(16) glm::mat4 mvp;
    alignas(16) glm::mat4 mvp_inv;
    alignas(16) glm::vec4 attr;
    alignas(16) glm::vec3 dir, up, right, pos;
    alignas(16) glm::vec3 light_dir;
    alignas(4) float near_;
    alignas(4) float far_;
    alignas(4) float p2_z;
    alignas(4) float p3_z;
    alignas(4) float frustum_ratio_x;
    alignas(4) float frustum_ratio_y;
    alignas(4) float frustum_ratio_w;
    alignas(4) float frustum_ratio_h;
};

using ubo_params_t = ubo_params;

struct param_package {
    glm::vec3 dir, up, right, pos;
    float rad;
    glm::vec3 light_dir;
    glm::vec4 global_col;
    glm::vec4 attr;
    glm::mat4 mvp;
    glm::mat4 mvp_inv;
    glm::mat4 mvp_trans;
    float near_;
    float far_;
};

using param_package_t = param_package;

enum class upload_mode { POS_COL_SEP, FULL_SEP, VEC3_SEP, NO_SEP, BUFFER_ARRAY, NULL_MODE };

using upload_mode_ut = std::underlying_type_t<upload_mode>;

static inline std::array<std::string, 6> upload_mode_string = {
    "POS_COL_SEP", "FULL_SEP", "VEC3_SEP", "NO_SEP", "BUFFER_ARRAY", "NULL_MODE"};

class rendering_task {
public:
    template<typename... Paths>
    rendering_task(upload_mode const& mode, std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options,
        Paths... paths)
            : mode_(mode) {
        try {
            program_ = core::utility::make_glowl_shader(label, options, std::forward<Paths>(paths)...);
        } catch (...) {
            core::utility::log::Log::DefaultLog.WriteError("[rendering_task] Failed to create program");
            throw;
        }
    }

    virtual ~rendering_task() = default;

    virtual bool render(GLuint ubo) = 0;

    virtual bool upload(data_package_t const& package) = 0;

    virtual bool cleanup() = 0;

    upload_mode get_mode() const {
        return mode_;
    }

protected:
    glowl::GLSLProgram* get_program() const {
        return program_.get();
    }

private:
    upload_mode mode_;

    std::unique_ptr<glowl::GLSLProgram> program_;
};
} // namespace megamol::test_gl::rendering
