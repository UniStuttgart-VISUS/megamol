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
#include <type_traits>

#include "msf/LineTranslator.h"
#include "msf/ShaderFactoryOptionsOpenGL.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::core::utility {

glowl::GLSLProgram::ShaderSourceList::value_type make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, msf::ShaderFactoryOptionsOpenGL const& options,
    msf::LineTranslator& translator);

namespace {
template<bool Shared>
using program_ptr_t =
    std::conditional_t<Shared, std::shared_ptr<glowl::GLSLProgram>, std::unique_ptr<glowl::GLSLProgram>>;

template<bool Shared, typename... Paths>
program_ptr_t<Shared> make_glowl_shader_impl(
    std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths) {
    msf::LineTranslator translator;
    try {
        program_ptr_t<Shared> program;
        if constexpr (Shared) {
            program = std::make_shared<glowl::GLSLProgram>(
                glowl::GLSLProgram::ShaderSourceList{make_glowl_shader_source(paths, options, translator)...});

        } else {
            program = std::make_unique<glowl::GLSLProgram>(
                glowl::GLSLProgram::ShaderSourceList{make_glowl_shader_source(paths, options, translator)...});
        }
        program->setDebugLabel(label);
        return program;
    } catch (glowl::GLSLProgramException const& ex) {
        throw glowl::GLSLProgramException(
            "Error building shader program \"" + label + "\":\n" + translator.translateErrorLog(ex.what()));
    }
}
} // namespace

template<typename... Paths>
std::enable_if_t<(std::is_convertible_v<Paths, std::filesystem::path> && ...), std::unique_ptr<glowl::GLSLProgram>>
make_glowl_shader(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths) {
    return make_glowl_shader_impl<false>(label, options, std::forward<Paths>(paths)...);
}

template<typename... Paths>
std::enable_if_t<(std::is_convertible_v<Paths, std::filesystem::path> && ...), std::shared_ptr<glowl::GLSLProgram>>
make_shared_glowl_shader(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths) {
    return make_glowl_shader_impl<true>(label, options, std::forward<Paths>(paths)...);
}

} // end namespace megamol::core::utility
