/*
 * shader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "glad/glad.h"

#include <string>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        namespace utility
        {
            /**
            * Create shader, additionally performing checks.
            * Throws an exception if it fails to compile the shader.
            *
            * @param shader Shader text
            * @param type Shader type
            *
            * @return Shader handle
            */
            GLuint make_shader(const std::string& shader, GLenum type);

            /**
            * Create program, additionally performing checks.
            * Throws an exception if it fails to link the shaders.
            *
            * @param shader Shader text
            * @param type Shader type
            *
            * @return Shader handle
            */
            GLuint make_program(const std::vector<GLuint>& shader_handles);
        }
    }
}