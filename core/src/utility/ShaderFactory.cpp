/*
 * ShaderFactory.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/ShaderFactory.h"


GLuint megamol::core::utility::make_shader(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options) {
    megamol::shaderfactory::compiler cp;
    auto shader_string = cp.preprocess(shader_source_path, options);

    auto shader = glCreateShader(megamol::shaderfactory::get_shader_type_ogl(shader_source_path));
    auto const shader_cstr = shader_string.c_str();
    GLint const shader_size = shader_string.size();
    glShaderSource(shader, 1, &shader_cstr, &shader_size);
    glCompileShader(shader);
    if (!check_and_log_shader_compile(shader)) {
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}
