/*
 * ShaderFactory.cpp
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/ShaderFactory.h"

#include "msf/compiler.h"
#include "msf/compiler_utils.h"


static megamol::shaderfactory::compiler msf_cp;


GLuint megamol::core::utility::make_shader(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options) {
    auto shader_string = msf_cp.preprocess(shader_source_path, options);

    auto shader = glCreateShader(megamol::shaderfactory::get_shader_type_ogl(shader_source_path));
    auto const shader_cstr = shader_string.c_str();
    GLint const shader_size = shader_string.size();
    glShaderSource(shader, 1, &shader_cstr, nullptr);
    glCompileShader(shader);
    if (!check_and_log_shader_compile(shader)) {
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}


glowl::GLSLProgram::ShaderSourceList::value_type megamol::core::utility::make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options) {
    auto const shader_string = msf_cp.preprocess(shader_source_path, options);
    auto const type = megamol::shaderfactory::get_shader_type_ogl(shader_source_path);

    return std::make_pair(static_cast<glowl::GLSLProgram::ShaderType>(type), shader_string);
}
