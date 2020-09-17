/*
 * GLSLShader.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#pragma once

#include <unordered_map>

#include "glad/glad.h"

#include "mmcore/utility/ShaderFactory.h"

namespace megamol::core::utility::graphics {

class GLSLShader {
public:
    struct glsl_uniform {
        GLint location;
        GLsizei count;
    };
    using glsl_uniform_t = glsl_uniform;

    GLSLShader() : _program(0) {}

    template <typename... Paths>
    GLSLShader(megamol::shaderfactory::compiler_options const& options, Paths... paths)
        : _program(megamol::core::utility::make_program(options, std::forward<Paths>(paths)...)) {
        // https://github.com/fendevel/Guide-to-Modern-OpenGL-Functions
        GLint uniform_count = 0;
        glGetProgramiv(_program, GL_ACTIVE_UNIFORMS, &uniform_count);

        if (uniform_count != 0) {
            GLint max_name_len = 0;
            glGetProgramiv(_program, GL_ACTIVE_UNIFORM_MAX_LENGTH, &max_name_len);

            auto name = std::make_unique<char[]>(max_name_len);

            GLsizei length = 0;
            GLsizei count = 0;
            GLenum type = GL_NONE;

            for (GLint idx = 0; idx < uniform_count; ++idx) {
                glGetActiveUniform(_program, idx, max_name_len, &length, &count, &type, name.get());

                _uniform_map.emplace(std::make_pair(std::string(name.get(), length),
                    glsl_uniform_t{glGetUniformLocation(_program, name.get()), count}));
            }
        }
    }

    GLint get_uniform_location(std::string const& name) { return _uniform_map[name].location; }

    GLsizei get_uniform_size(std::string const& name) { return _uniform_map[name].count; }

    void enable() const { glUseProgram(_program); }

    void disable() const { glUseProgram(0); }

    ~GLSLShader() { glDeleteShader(_program); }

    operator GLuint() { return _program; }

private:
    GLuint _program;

    std::unordered_map<std::string, glsl_uniform_t> _uniform_map;
};

} // namespace megamol::core::utility::graphics