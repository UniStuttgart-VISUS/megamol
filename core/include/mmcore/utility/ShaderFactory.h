/*
 * ShaderFactory.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

#include "glad/glad.h"

#include "msf/compiler.h"
#include "msf/compiler_options.h"

#include "mmcore/utility/log/Log.h"

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

        megamol::core::utility::log::Log::DefaultLog.WriteError("MSF Shader Compile Error:\n%s", shader_info);
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

        megamol::core::utility::log::Log::DefaultLog.WriteError("MSF Program Link Error:\n%s", program_info);
    }

    return success;
}


inline void add_vendor_definition(megamol::shaderfactory::compiler_options& options) {
    std::basic_string<GLubyte> const tmp(glGetString(GL_VENDOR));
    std::string vendor_str;
    vendor_str.resize(tmp.size());
    std::transform(tmp.cbegin(), tmp.cend(), vendor_str.begin(), [](GLubyte chr) { return std::tolower(chr); });

    if (vendor_str.find("ati") != std::string::npos || vendor_str.find("amd") != std::string::npos) {
        options.add_definition("__AMD__");
    }

    if (vendor_str.find("nvidia") != std::string::npos) {
        options.add_definition("__NVIDIA__");
    }

    if (vendor_str.find("intel") != std::string::npos) {
        options.add_definition("__INTEL__");
    }
}


GLuint make_shader(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options);


template <typename... Paths>
std::enable_if_t<(std::is_convertible_v<std::filesystem::path, Paths> && ...), GLuint> make_program(
    megamol::shaderfactory::compiler_options const& options, Paths... paths) {
    megamol::shaderfactory::compiler_options tmp_opt = options;
    add_vendor_definition(tmp_opt);
    constexpr std::size_t size = sizeof...(paths);

    GLuint shaders[size] = {make_shader(paths, tmp_opt)...};

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
    }

    return program;
}

} // end namespace megamol::core::utility
