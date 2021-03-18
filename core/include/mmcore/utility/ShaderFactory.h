/*
 * ShaderFactory.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

#include "msf/compiler_options.h"

#include "glad/glad.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::core::utility {

glowl::GLSLProgram::ShaderSourceList::value_type make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, megamol::shaderfactory::compiler_options const& options);


template<typename... Paths>
std::enable_if_t<(std::is_convertible_v<std::filesystem::path, Paths> && ...), glowl::GLSLProgram::ShaderSourceList>
make_glowl_shader_source_list(megamol::shaderfactory::compiler_options const& options, Paths... paths) {
    return {make_glowl_shader_source(paths, options)...};
}

} // end namespace megamol::core::utility
