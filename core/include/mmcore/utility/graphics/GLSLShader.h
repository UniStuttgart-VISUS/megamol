/*
 * GLSLShader.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#pragma once

#include <unordered_map>

#include "glad/glad.h"

#include "mmcore/utility/ShaderFactory.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::core::utility::graphics {

class GLSLShader : public glowl::GLSLProgram {
public:
    template<typename... Paths>
    GLSLShader(megamol::shaderfactory::compiler_options const& options, Paths... paths)
            : glowl::GLSLProgram(
                  megamol::core::utility::make_glowl_shader_source_list(options, std::forward<Paths>(paths)...)) {}
};

} // namespace megamol::core::utility::graphics
