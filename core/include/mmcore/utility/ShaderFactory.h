/*
 * ShaderFactory.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>

#include "msf/ShaderFactoryOptionsOpenGL.h"

#include "glad/glad.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::core::utility {

std::unique_ptr<glowl::GLSLProgram> make_glowl_shader(std::string const& label,
    msf::ShaderFactoryOptionsOpenGL const& options, const std::vector<std::filesystem::path>& paths);

template<typename... Paths>
std::enable_if_t<(std::is_convertible_v<Paths, std::filesystem::path> && ...), std::unique_ptr<glowl::GLSLProgram>>
make_glowl_shader(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths) {
    return make_glowl_shader(label, options, std::vector<std::filesystem::path>{paths...});
}

} // end namespace megamol::core::utility
