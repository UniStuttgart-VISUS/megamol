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

#include "msf/LineTranslator.h"
#include "msf/ShaderFactoryOptionsOpenGL.h"

#include "glad/glad.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::core::utility {

glowl::GLSLProgram::ShaderSourceList::value_type make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, msf::ShaderFactoryOptionsOpenGL const& options,
    msf::LineTranslator& translator);

template<typename... Paths>
std::enable_if_t<(std::is_convertible_v<Paths, std::filesystem::path> && ...), std::unique_ptr<glowl::GLSLProgram>>
make_glowl_shader(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths) {
    msf::LineTranslator translator;
    try {
        auto program = std::make_unique<glowl::GLSLProgram>(
            glowl::GLSLProgram::ShaderSourceList{make_glowl_shader_source(paths, options, translator)...});
        program->setDebugLabel(label);
        return program;
    } catch (glowl::GLSLProgramException const& ex) {
        throw glowl::GLSLProgramException(translator.translateErrorLog(ex.what()));
    }
}

} // end namespace megamol::core::utility
