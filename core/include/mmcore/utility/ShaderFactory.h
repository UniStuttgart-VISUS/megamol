#pragma once

#include <filesystem>
#include <string>

#include "glad/glad.h"

#include "msf/compiler.h"
#include "msf/compiler_options.h"

#include "mmcore/utility/log/Log.h"

namespace megamol::core::utility {

GLint check_and_log_shader_compile(GLuint shader) {
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        GLint logSize = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
        std::string shader_info;
        shader_info.resize(logSize);
        glGetShaderInfoLog(shader, logSize, &logSize, shader_info.data());

        megamol::core::utility::log::Log::DefaultLog.WriteError("MSF Shader Compile Error\n%s", shader_info);
    }

    return success;
}

GLuint make_shader(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options);

} // end namespace megamol::core::utility
