/*
 * ShaderFactory.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

#include "glad/glad.h"

#include "msf/compiler_options.h"

#include "mmcore/utility/log/Log.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::core::utility {

inline GLint check_and_log_shader_compile(GLuint shader) {
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        GLint logSize = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
        std::string shader_info;
        shader_info.resize(logSize);
        glGetShaderInfoLog(shader, logSize, &logSize, shader_info.data());

        megamol::core::utility::log::Log::DefaultLog.WriteError("MSF Shader Compile Error:\n%s", shader_info.data());
    }

    return success;
}


inline GLint check_and_log_program_link(GLuint program) {
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        GLint logSize = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
        std::string program_info;
        program_info.resize(logSize);
        glGetProgramInfoLog(program, logSize, &logSize, program_info.data());

        megamol::core::utility::log::Log::DefaultLog.WriteError("MSF Program Link Error:\n%s", program_info.data());
    }

    return success;
}


GLuint make_shader(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options);


template <typename... Paths>
std::enable_if_t<(std::is_convertible_v<std::filesystem::path, Paths> && ...), GLuint> make_program(
    megamol::shaderfactory::compiler_options const& options, Paths... paths) {
    constexpr std::size_t size = sizeof...(paths);

    GLuint shaders[size] = {make_shader(paths, options)...};

    auto program = glCreateProgram();

    for (std::size_t idx = 0; idx < size; ++idx) {
        glAttachShader(program, shaders[idx]);
    }

    glLinkProgram(program);

    if (!check_and_log_program_link(program)) {
        glDeleteProgram(program);

        for (std::size_t idx = 0; idx < size; ++idx) {
            glDeleteShader(shaders[idx]);
        }

        return 0;
    }

    for (std::size_t idx = 0; idx < size; ++idx) {
        glDetachShader(program, shaders[idx]);
        glDeleteShader(shaders[idx]);
    }

    return program;
}


glowl::GLSLProgram::ShaderSourceList::value_type make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options);


template<typename... Paths>
std::enable_if_t<(std::is_convertible_v<std::filesystem::path, Paths> && ...), glowl::GLSLProgram::ShaderSourceList> make_glowl_shader_source_list(
    megamol::shaderfactory::compiler_options const& options, Paths... paths) {
    return {make_glowl_shader_source(paths, options)...};
}

} // end namespace megamol::core::utility
